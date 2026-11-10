# -*- coding: utf-8 -*-
"""
step8_retrain_test.py - 재학습 주기 단축에 의한 오차 저감 실증 (과제 C-3)
==============================================================================
배경 (2026.9.3 운영 성능 분석)
  · 2026년 9월 운영 MAE 13.22원 / rMAE 1.311 - 8월(8.16 / 0.731) 대비 급락
  · 주원인은 SMP 수준 급등(실적평균 142.5 -> 162.6원)에 따른 과소예측
  · 현행 재학습 주기는 **주 1회(일요일 02:00)** 이며, 수준 전환기 대응 지연 가능성
  · 과제 C-3(재학습 주기 단축 검토)의 기한을 2026.12 -> 10월로 앞당겨 실증

설계
  · 홀드아웃 구간을 순회하며 **주기별로 학습 종료일을 갱신**하고 D+1 예측을 산출
  · 각 예측일의 학습 구간 = TRAIN_START ~ (직전 재학습 시점)
  · 주기 : 고정(재학습 없음) · 28일 · 14일 · 7일
  · 재학습 소요시간을 함께 측정하여 **운영 부하** 판단 근거로 제시

판정
  · 주지표 MAE·rMAE 개선폭과 재학습 부하(회/월·소요초)를 병기
  · 급등 구간(2026-08~) 개선 여부를 별도 확인 - 수준 전환 대응력의 직접 지표

유의
  · 학습 종료일 이후 정보는 일절 사용하지 않음 - 누출 차단
  · 운영 모델을 갱신하지 않음(실증 전용)

산출
  · logs/step8_retrain_test_{타깃}_{일시}.csv

상호작용 실증 (2x2)
  · 재학습 주기가 수준 전환을 이미 흡수한다면 수준 편차 변수의 순효과는 축소됨
  · --level both 지정 시 동일 주기에서 기준·수준변수 구성을 모두 산출하여 순효과 측정

실행
  python scripts\\step8_retrain_test.py
  python scripts\\step8_retrain_test.py --start 2026-04-01     (구간 단축 - 소요 절감)
  python scripts\\step8_retrain_test.py --cycles 28,14         (주기 한정)

버전 : v1.0 (2026-09-03)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step8_level_test import add_level, NEW as LEVEL_NEW  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MAPE_FLOOR = 20.0
SEED = 42
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju", "year",
           "unit_id", "unit_name", "grp", "pv_source", "holiday_name"}
TRAIN_START = "2021-02-01"
SURGE_START = "2026-08-01"

_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {str(값):<26} {비고}", flush=True)
    _log.append({"항목": 항목, "값": str(값), "비고": 비고})


def mape(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.mean(np.abs(p[m] - a[m]) / np.clip(a[m], MAPE_FLOOR, None)) * 100)


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean())


def make_xy(d, feats, t):
    X = d[feats].copy()
    for c in feats:
        if c in ("month", "dow", "hour_n", "daytype", "season") \
           or X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category")
    return X, d[t]


def fit(tr, feats, t):
    X, y = make_xy(tr, feats, t)
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          min_child_samples=30, subsample=0.9,
                          colsample_bytree=0.9, random_state=SEED, verbose=-1)
    m.fit(X, y)
    return m


def main():
    ap = argparse.ArgumentParser(description="재학습 주기 실증")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--start", default="2026-01-01", help="홀드아웃 시작일")
    ap.add_argument("--cycles", default="0,28,14,7",
                    help="재학습 주기(일) 쉼표 구분. 0 = 고정(재학습 없음)")
    ap.add_argument("--level", default="off", choices=["off", "on", "both"],
                    help="수준 편차 변수 투입 — off(기준) · on(투입) · "
                         "both(2x2 상호작용 실증). 기본 off")
    args = ap.parse_args()
    t = args.target
    cycles = [int(x) for x in args.cycles.split(",") if x.strip()]

    print("=" * 74)
    print(" step8_retrain_test - 재학습 주기 단축 실증 v1.0 (과제 C-3)")
    print(f" 실행시각 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 74)

    src = os.path.join(PROCESSED, f"smp_features_{t}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}.csv")
    for p in (src, dic):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 -> {p}")
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    dd = pd.read_csv(dic, encoding="utf-8-sig")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    feats = [c for c in feats if c in ("daytype", "season")
             or pd.api.types.is_numeric_dtype(df[c])]

    df = df.dropna(subset=[t]).sort_values(["date", "hour_n"]).reset_index(drop=True)

    #  수준 편차 변수 — step8_level_test 와 동일 정의를 재사용하여 조건 일치 보장
    variants = [("기준", feats)]
    if args.level in ("on", "both"):
        df = add_level(df, t)
        lv = [c for c in LEVEL_NEW if c in df.columns and df[c].notna().any()]
        if args.level == "on":
            variants = [("수준변수", feats + lv)]
        else:
            variants = [("기준", feats), ("수준변수", feats + lv)]

    ho = df[df["date"] >= args.start]
    days = sorted(ho["date"].unique())

    print("\n■ ① 구성")
    rep("투입 변수", f"{len(feats)}개")
    rep("홀드아웃", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(ho):,}행")
    rep("검증 주기", " · ".join("고정" if c == 0 else f"{c}일" for c in cycles))
    rep("변수 구성", " · ".join(f"{n}({len(f)}개)" for n, f in variants),
        "2x2 상호작용 실증" if args.level == "both" else "")

    print("\n■ ② 주기별 순회 학습·예측")
    print(f"  {'구성':<8}{'주기':<7}{'재학습':>7}{'학습초':>8}{'전체MAE':>9}{'rMAE':>8}"
          f"{'급등MAE':>9}{'MAPE':>8}")
    print("  " + "-" * 70)

    summary = []
    for vname, feats in variants:
      for cyc in cycles:
        preds, acts, dates, navs, n_fit, sec = [], [], [], [], 0, 0.0
        model, last_cut = None, None
        for d in days:
            cut = d if cyc == 0 else None
            if cyc == 0:
                need = model is None
                cut = args.start          # 고정 - 홀드아웃 직전까지만 학습
            else:
                if last_cut is None:
                    need = True
                else:
                    gap = (pd.to_datetime(d) - pd.to_datetime(last_cut)).days
                    need = gap >= cyc
                cut = d
            if need:
                tr = df[(df["date"] >= TRAIN_START) & (df["date"] < cut)]
                if len(tr) < 5000:
                    continue
                t0 = time.time()
                model = fit(tr, feats, t)
                sec += time.time() - t0
                n_fit += 1
                last_cut = d
            if model is None:
                continue
            seg = ho[ho["date"] == d]
            X, _ = make_xy(seg, feats, t)
            preds.append(model.predict(X))
            acts.append(seg[t].values)
            dates.append(np.repeat(d, len(seg)))
            navs.append(seg["smp_lag_d1"].values if "smp_lag_d1" in seg.columns
                        else np.full(len(seg), np.nan))

        p = np.concatenate(preds); a = np.concatenate(acts)
        dt = np.concatenate(dates)
        nv = np.concatenate(navs)
        nv = nv if np.isfinite(nv).any() else None
        sgi = dt >= SURGE_START
        row = {"변수구성": vname, "주기": "고정" if cyc == 0 else f"{cyc}일",
               "재학습횟수": n_fit, "학습소요초": round(sec, 1),
               "전체MAE": round(mae(p, a), 2),
               "rMAE": round(mae(p, a) / mae(nv, a), 3) if nv is not None else None,
               "급등MAE": round(mae(p[sgi], a[sgi]), 2) if sgi.any() else None,
               "MAPE": round(mape(p, a), 2),
               "편의": round(float(((p - a) / np.maximum(a, MAPE_FLOOR)).mean() * 100), 2)}
        summary.append(row)
        print(f"  {vname:<8}{row['주기']:<7}{n_fit:>7}{sec:>8.0f}{row['전체MAE']:>9.2f}"
              f"{(row['rMAE'] or 0):>8.3f}{(row['급등MAE'] or 0):>9.2f}{row['MAPE']:>7.2f}%")
        rep(f"{vname} 주기 {row['주기']}",
            f"MAE {row['전체MAE']} / rMAE {row['rMAE']}",
            f"급등 {row['급등MAE']} / 재학습 {n_fit}회 / {sec:.0f}초")

    print("  " + "-" * 70)

    print("\n■ ③ 판정")
    s = pd.DataFrame(summary)
    for vname, _ in variants:
        sv = s[s["변수구성"] == vname]
        b = sv[sv["주기"] == "고정"]
        if b.empty:
            continue
        b0 = float(b["전체MAE"].iloc[0]); g0 = float(b["급등MAE"].iloc[0])
        print(f"    [{vname}] 고정 대비")
        for _, r in sv.iterrows():
            if r["주기"] == "고정":
                continue
            print(f"      {r['주기']:<6} 전체 {b0:.2f} -> {r['전체MAE']:.2f}"
                  f" ({r['전체MAE']-b0:+.2f})   급등 {g0:.2f} -> {r['급등MAE']:.2f}"
                  f" ({r['급등MAE']-g0:+.2f})   부하 {r['재학습횟수']}회")
            rep(f"{vname} 개선 {r['주기']}", f"전체 {r['전체MAE']-b0:+.2f}원",
                f"급등 {r['급등MAE']-g0:+.2f}원 / 재학습 {r['재학습횟수']}회")

    #  상호작용 — 동일 주기에서 수준변수 순효과
    if len(variants) == 2:
        print("\n    [상호작용] 동일 주기에서 수준변수 투입의 순효과")
        a = s[s["변수구성"] == "기준"].set_index("주기")
        c = s[s["변수구성"] == "수준변수"].set_index("주기")
        for k in a.index:
            if k not in c.index:
                continue
            d1 = float(c.loc[k, "전체MAE"]) - float(a.loc[k, "전체MAE"])
            d2 = float(c.loc[k, "급등MAE"]) - float(a.loc[k, "급등MAE"])
            print(f"      {k:<6} 전체 {a.loc[k,'전체MAE']:.2f} -> "
                  f"{c.loc[k,'전체MAE']:.2f} ({d1:+.2f})   "
                  f"급등 {a.loc[k,'급등MAE']:.2f} -> {c.loc[k,'급등MAE']:.2f} ({d2:+.2f})")
            rep(f"수준변수 순효과 {k}", f"전체 {d1:+.2f}원", f"급등 {d2:+.2f}원")
        print("\n    ※ 재학습 주기가 짧아질수록 순효과가 축소되면,"
              " 수준변수의 정보를 재학습이 흡수하는 것으로 해석")
    print("\n    ※ 채택 기준 : 전체·급등 MAE 동시 개선 + 재학습 부하가 운영 스케줄 내")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_retrain_test_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_log).to_csv(f, index=False)
        f.write("\n")
        s.to_csv(f, index=False)
    print(f"\n[완료] 로그 -> {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
