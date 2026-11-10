# -*- coding: utf-8 -*-
"""
step8_holiday_train_test.py — 학습 단계 휴일 대응력 개선 실증 (v1.0)
==============================================================================
배경
  · 2026.9.6 요일·휴일 실증에서 **모형 자체의 휴일 과대예측** 확인
    | 구분 | 일수 | MAE(무보정) | MAE(보정후) | 편의 |
    |------|-----:|------------:|------------:|-----:|
    | 평일 | 18 | 11.16 | 7.85 | +0.10% |
    | 일·공휴일 | 4 | 14.97 | 14.32 | +13.33% |
    | 토 | 2 | 24.64 | 23.32 | +15.72% |
  · 편의 보정 2분류 분리로 전체 −9.8% 개선하였으나, 휴일 보정계수가
    상한 ±25원에 포화(12~16시 잔차 −21~−28원) — **사후 보정의 한계**
  · 학습 변수에는 휴일 특성 12종이 이미 투입되어 있으므로,
    남은 경로는 **학습 가중·구조**의 조정

가설
  ① 휴일 표본 가중(sample_weight) 상향 시 휴일 오차가 축소됨
  ② 평일·휴일 분리 모델이 단일 모델보다 휴일 구간에서 우월함

비교 구성
  | No | 구성 | 내용 |
  |----|------|------|
  | ① | 기준 | 현행 — 전 표본 동일 가중, 단일 모델 |
  | ② | 가중 w2 | 휴일 표본 가중 2배 |
  | ③ | 가중 w3 | 휴일 표본 가중 3배 |
  | ④ | 가중 w5 | 휴일 표본 가중 5배 |
  | ⑤ | 분리 모델 | 평일·휴일 각각 학습 후 구분별 적용 |

설계 원칙
  · **운영과 동일한 재학습 조건**(주 1회, 7일 주기 순회)에서 평가
    — 2026.9.4 수준 편차 변수 실증에서, 고정 학습 조건의 개선이
      운영 조건에서 소멸한 사례를 반영
  · 학습 종료일 이후 정보 미사용 — 누출 차단
  · 운영 모델 미갱신(실증 전용)

판정
  · 채택 기준 : **휴일 MAE 개선 + 평일 MAE 비악화**
  · 휴일 구간을 토·일공휴로 분해하여 병기

실행
  python scripts\\step8_holiday_train_test.py
  python scripts\\step8_holiday_train_test.py --start 2026-05-01     (구간 단축)
  python scripts\\step8_holiday_train_test.py --schemes 기준,w3,분리

버전 : v1.0 (2026-09-06)
"""
import os
import sys
import time
import argparse
import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("[중단] lightgbm 미설치")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MAPE_FLOOR = 20.0
SEED = 42
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju", "year",
           "unit_id", "unit_name", "grp", "pv_source", "holiday_name"}
TRAIN_START = "2021-02-01"
CYCLE = 7                       # 운영 재학습 주기(일) — 일요일 02:00 주 1회
_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {str(값):<26} {비고}", flush=True)
    _log.append({"항목": 항목, "값": str(값), "비고": 비고})


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def mape(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.mean(np.abs(p[m] - a[m]) / np.clip(a[m], MAPE_FLOOR, None)) * 100)


def make_xy(d, feats, t):
    X = d[feats].copy()
    for c in feats:
        if c in ("month", "dow", "hour_n", "daytype", "season") \
           or X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category")
    return X, d[t]


def fit(tr, feats, t, w=None):
    X, y = make_xy(tr, feats, t)
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          min_child_samples=30, subsample=0.9,
                          colsample_bytree=0.9, random_state=SEED, verbose=-1)
    m.fit(X, y, sample_weight=w)
    return m


def holiday_mask(d):
    """휴일 여부 — daytype 우선, 부재 시 dow·is_holiday 조합"""
    if "daytype" in d.columns:
        return (d["daytype"].astype(str) == "휴일").to_numpy()
    off = (d["dow"].astype(int) >= 5)
    if "is_holiday" in d.columns:
        off = off | (d["is_holiday"].fillna(0).astype(int) == 1)
    return off.to_numpy()


def group3(d):
    """평일 · 토 · 일공휴 3분류 — 성능 분해용"""
    dow = d["dow"].astype(int).to_numpy()
    hol = (d["is_holiday"].fillna(0).astype(int).to_numpy()
           if "is_holiday" in d.columns else np.zeros(len(d), int))
    g = np.full(len(d), "평일", dtype=object)
    g[dow == 5] = "토"
    g[(dow == 6) | (hol == 1)] = "일공휴"
    return g


def main():
    ap = argparse.ArgumentParser(description="학습 단계 휴일 대응력 개선 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--start", default="2026-03-01", help="홀드아웃 시작일")
    ap.add_argument("--cycle", type=int, default=CYCLE, help="재학습 주기(일)")
    ap.add_argument("--schemes", default="기준,w2,w3,w5,분리",
                    help="비교 구성(쉼표 구분) — 기준·w2·w3·w5·분리")
    args = ap.parse_args()
    t = args.target
    schemes = [x.strip() for x in args.schemes.split(",") if x.strip()]

    print("=" * 78)
    print(" step8_holiday_train_test — 학습 단계 휴일 대응력 개선 실증 v1.0")
    print(f" 실행시각 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    src = os.path.join(PROCESSED, f"smp_features_{t}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}.csv")
    for p in (src, dic):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    dd = pd.read_csv(dic, encoding="utf-8-sig")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    feats = [c for c in feats if c in ("daytype", "season")
             or pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=[t]).sort_values(["date", "hour_n"]).reset_index(drop=True)

    ho = df[df["date"] >= args.start]
    days = sorted(ho["date"].unique())
    if not days:
        sys.exit("[중단] 홀드아웃 구간 없음")

    g_all = group3(ho)
    print("\n■ ① 구성")
    rep("투입 변수", f"{len(feats)}개")
    rep("홀드아웃", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(ho):,}행")
    rep("재학습 주기", f"{args.cycle}일", "운영 조건과 동일(주 1회)")
    rep("휴일 표본 비중",
        f"{holiday_mask(ho).mean() * 100:.1f}%",
        f"평일 {(g_all == '평일').sum():,}행 / 토 {(g_all == '토').sum():,}행 / "
        f"일공휴 {(g_all == '일공휴').sum():,}행")
    rep("비교 구성", " · ".join(schemes))

    print("\n■ ② 구성별 순회 학습·예측")
    print(f"  {'구성':<8}{'재학습':>7}{'학습초':>8}{'전체MAE':>9}{'평일':>8}"
          f"{'토':>8}{'일공휴':>8}{'rMAE':>8}{'휴일편의%':>10}")
    print("  " + "-" * 76)

    summary = []
    for sc in schemes:
        preds, acts, dates, navs, n_fit, sec = [], [], [], [], 0, 0.0
        model = None            # 단일 모델 또는 (평일모델, 휴일모델)
        last_cut = None
        for d in days:
            gap = None if last_cut is None else \
                (pd.to_datetime(d) - pd.to_datetime(last_cut)).days
            if last_cut is None or gap >= args.cycle:
                tr = df[(df["date"] >= TRAIN_START) & (df["date"] < d)]
                if len(tr) < 5000:
                    continue
                t0 = time.time()
                if sc == "분리":
                    hm = holiday_mask(tr)
                    model = (fit(tr[~hm], feats, t), fit(tr[hm], feats, t))
                else:
                    w = None
                    if sc.startswith("w"):
                        k = float(sc[1:])
                        w = np.where(holiday_mask(tr), k, 1.0)
                    model = fit(tr, feats, t, w)
                sec += time.time() - t0
                n_fit += 1
                last_cut = d
            if model is None:
                continue
            seg = ho[ho["date"] == d]
            X, _ = make_xy(seg, feats, t)
            if sc == "분리":
                p = (model[1] if holiday_mask(seg)[0] else model[0]).predict(X)
            else:
                p = model.predict(X)
            preds.append(p)
            acts.append(seg[t].values)
            dates.append(np.repeat(d, len(seg)))
            navs.append(seg["smp_lag_d1"].values if "smp_lag_d1" in seg.columns
                        else np.full(len(seg), np.nan))

        if not preds:
            print(f"  {sc:<8} — 평가 표본 없음")
            continue
        p = np.concatenate(preds)
        a = np.concatenate(acts)
        dt = np.concatenate(dates)
        nv = np.concatenate(navs)
        nv = nv if np.isfinite(nv).any() else None
        g = group3(ho[ho["date"].isin(np.unique(dt))])
        hol = g != "평일"
        row = {"구성": sc, "재학습": n_fit, "학습초": round(sec, 1),
               "전체MAE": round(mae(p, a), 2),
               "평일": round(mae(p[g == "평일"], a[g == "평일"]), 2),
               "토": round(mae(p[g == "토"], a[g == "토"]), 2),
               "일공휴": round(mae(p[g == "일공휴"], a[g == "일공휴"]), 2),
               "휴일MAE": round(mae(p[hol], a[hol]), 2),
               "rMAE": round(mae(p, a) / mae(nv, a), 3) if nv is not None else None,
               "MAPE": round(mape(p, a), 2),
               "휴일편의": round(float(((p[hol] - a[hol]) /
                                    np.maximum(a[hol], MAPE_FLOOR)).mean() * 100), 2)}
        summary.append(row)
        print(f"  {sc:<8}{n_fit:>7}{sec:>8.0f}{row['전체MAE']:>9.2f}{row['평일']:>8.2f}"
              f"{row['토']:>8.2f}{row['일공휴']:>8.2f}{(row['rMAE'] or 0):>8.3f}"
              f"{row['휴일편의']:>10.2f}")
        rep(f"구성 {sc}", f"MAE {row['전체MAE']} / rMAE {row['rMAE']}",
            f"평일 {row['평일']} / 토 {row['토']} / 일공휴 {row['일공휴']} / "
            f"휴일편의 {row['휴일편의']}%")
    print("  " + "-" * 76)

    if not summary:
        sys.exit("[중단] 결과 없음")
    s = pd.DataFrame(summary).set_index("구성")
    if "기준" not in s.index:
        print("\n[주의] 기준 구성 미포함 — 상대 판정 생략")
        return

    b = s.loc["기준"]
    print("\n■ ③ 판정 — 기준 대비")
    for nm in s.index:
        if nm == "기준":
            continue
        print(f"    {nm:<8} 전체 {s.loc[nm, '전체MAE'] - b['전체MAE']:+.2f}원 / "
              f"평일 {s.loc[nm, '평일'] - b['평일']:+.2f} / "
              f"토 {s.loc[nm, '토'] - b['토']:+.2f} / "
              f"일공휴 {s.loc[nm, '일공휴'] - b['일공휴']:+.2f} / "
              f"휴일편의 {s.loc[nm, '휴일편의'] - b['휴일편의']:+.2f}%p")
        rep(f"기준 대비 {nm}", f"전체 {s.loc[nm, '전체MAE'] - b['전체MAE']:+.2f}원",
            f"휴일 {s.loc[nm, '휴일MAE'] - b['휴일MAE']:+.2f}원 / "
            f"평일 {s.loc[nm, '평일'] - b['평일']:+.2f}원")
    best = s["휴일MAE"].idxmin()
    print(f"\n    휴일 MAE 최소 : {best}")
    rep("휴일 최우수 구성", best, "휴일 MAE 기준")
    print("    ※ 채택 기준 : 휴일 MAE 개선 + 평일 MAE 비악화")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_holiday_train_test_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_log).to_csv(f, index=False)
        f.write("\n")
        s.reset_index().to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
