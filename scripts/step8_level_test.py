# -*- coding: utf-8 -*-
"""
step8_level_test.py - 수준 편차 변수 투입에 의한 오차 저감 실증
==============================================================================
배경 (2026.9.3 운영 성능 분석)
  · 2026년 9월 운영 실적이 8월 대비 급락 - MAE 8.16 -> 13.22원, rMAE 0.731 -> 1.311
  · 원인 분석 결과 **SMP 수준 급등**(실적평균 142.5 -> 162.6원, +14.1%)에 따른
    체계적 과소예측이 주원인으로 확인됨(9.1 편의 −6.20%)
  · 편의 보정은 전 일자에서 오차를 축소하여 무혐의 - 보정 전 대비 -0.31~-2.12원
  · 트리 모델은 학습 분포 밖 수준을 외삽하지 못하므로,
    **수준 전환을 신호로 제공하는 변수**의 유효성을 검증할 필요

가설
  · 최근 단기(3일) 수준과 중기(7·30일) 수준의 괴리를 변수로 제공하면
    수준 전환 국면에서 과소·과대예측이 완화됨

신설 변수 (전량 D-1까지 확보 가능 - 누출 없음)
  | 변수 | 정의 |
  |------|------|
  | smp_ma3            | 동시간 기준 직전 3일 평균 |
  | smp_dev_ma3        | smp_lag_d1 - smp_ma3 - 단기 이탈폭 |
  | smp_ratio_ma3_ma7  | smp_ma3 / smp_ma7 - 단기·중기 수준비 |
  | smp_ratio_ma3_ma30 | smp_ma3 / smp_ma30 - 수준 전환 강도 |
  | smp_daily_ma3      | 직전 3일 일평균 |
  | smp_level_accel    | (lag_d1 - lag_d2) - (lag_d2 - lag_d3) - 수준 변화 가속 |

기존 보유 변수 : smp_ma7 · smp_ma30 · smp_ewm7 · smp_same_hour_ma7 ·
                smp_trend_d1_w1 · smp_ratio_d1_w1 (단기 3일 계열은 부재)

판정 (G-1 준용)
  · 전체 MAE 개선 + **급등구간(2026-08~) MAE 개선**, 부호 일관 시 채택
  · MAPE·rMAE 병기 (목표오차 확정서 v1.0)

산출
  · logs/step8_level_test_{일시}.csv

실행
  python scripts\\step8_level_test.py                    (신설 6종 전량)
  python scripts\\step8_level_test.py --vars smp_dev_ma3,smp_level_accel,smp_ratio_ma3_ma7,smp_ratio_ma3_ma30
                                                          (축약안 4종 — 중요도 상위)
  python scripts\\step8_level_test.py --target smp_jeju

버전 : v1.0 (2026-09-03)
"""
import os
import sys
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
CV_END = "2025-12-31"
HOLD_START = "2026-01-01"
SURGE_START = "2026-08-01"          # 급등 구간 - 9월 성능 급락의 발원
NEW = ["smp_ma3", "smp_dev_ma3", "smp_ratio_ma3_ma7",
       "smp_ratio_ma3_ma30", "smp_daily_ma3", "smp_level_accel"]

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


def add_level(df, t):
    """수준 편차 변수 생성 - 기존 피처 재사용, 부재 시에만 보강"""
    d = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    g = d.groupby("hour_n", sort=False)[t]
    l1 = d["smp_lag_d1"] if "smp_lag_d1" in d.columns else g.shift(24)
    l2 = d["smp_lag_d2"] if "smp_lag_d2" in d.columns else g.shift(48)
    l3 = d["smp_lag_d3"] if "smp_lag_d3" in d.columns else g.shift(72)
    if "smp_ma3" not in d.columns:
        d["smp_ma3"] = (l1 + l2 + l3) / 3.0
    if "smp_dev_ma3" not in d.columns:
        d["smp_dev_ma3"] = l1 - d["smp_ma3"]
    for c, base in (("smp_ratio_ma3_ma7", "smp_ma7"),
                    ("smp_ratio_ma3_ma30", "smp_ma30")):
        if c not in d.columns:
            d[c] = d["smp_ma3"] / d[base].replace(0, np.nan) if base in d.columns else np.nan
    if "smp_daily_ma3" not in d.columns:
        daily = (d[["date", "smp_lag_d1_dmean"]].drop_duplicates("date").sort_values("date")
                 if "smp_lag_d1_dmean" in d.columns else None)
        if daily is not None:
            daily["smp_daily_ma3"] = daily["smp_lag_d1_dmean"].shift(1).rolling(3, min_periods=2).mean()
            d = d.merge(daily[["date", "smp_daily_ma3"]], on="date", how="left")
    if "smp_level_accel" not in d.columns:
        d["smp_level_accel"] = (l1 - l2) - (l2 - l3)
    return d


def fit_predict(tr, te, feats, t):
    cats = [c for c in ("month", "dow", "hour_n", "daytype", "season") if c in feats]
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    for c in feats:
        if c in cats or Xtr[c].dtype == object or str(Xtr[c].dtype) == "string":
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          min_child_samples=30, subsample=0.9,
                          colsample_bytree=0.9, random_state=SEED, verbose=-1)
    m.fit(Xtr, tr[t])
    return m.predict(Xte), m


def block(name, pred, ho, t):
    a = ho[t].values
    nv = ho["smp_lag_d1"].values if "smp_lag_d1" in ho.columns else None
    r = {"구간": name, "표본": len(ho), "MAE": round(mae(pred, a), 2),
         "MAPE": round(mape(pred, a), 2),
         "편의": round(float(((pred - a) / np.maximum(a, MAPE_FLOOR)).mean() * 100), 2)}
    if nv is not None:
        r["rMAE"] = round(mae(pred, a) / mae(nv, a), 3)
    return r


def main():
    ap = argparse.ArgumentParser(description="수준 편차 변수 실증")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--vars", default="",
                    help="투입할 신설 변수 한정(쉼표 구분). 미지정 시 6종 전량 "
                         "— 축약안 실증용(예: smp_dev_ma3,smp_level_accel,"
                         "smp_ratio_ma3_ma7,smp_ratio_ma3_ma30)")
    args = ap.parse_args()
    t = args.target
    sel = [x.strip() for x in args.vars.split(",") if x.strip()] or None

    print("=" * 74)
    print(" step8_level_test - 수준 편차 변수 실증 v1.0")
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

    print("\n■ ① 변수 구성")
    base = [c for c in df.columns if c in ok and c not in EXCLUDE]
    base = [c for c in base if c in ("daytype", "season")
            or pd.api.types.is_numeric_dtype(df[c])]
    base = [c for c in base if c not in NEW]
    rep("기준 변수", f"{len(base)}개")

    df = add_level(df, t)
    cand = NEW if sel is None else [c for c in NEW if c in sel]
    if sel is not None:
        unknown = [c for c in sel if c not in NEW]
        if unknown:
            sys.exit(f"[중단] 정의되지 않은 변수 → {', '.join(unknown)}")
        rep("투입 한정", f"{len(cand)}개 / 정의 {len(NEW)}개", "축약안 실증")
    new = [c for c in cand if c in df.columns and df[c].notna().any()]
    rep("신설 수준 변수", f"{len(new)}개", " · ".join(new))
    miss = [c for c in cand if c not in new]
    if miss:
        rep("생성 실패", f"{len(miss)}개", " · ".join(miss))

    tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].dropna(subset=[t])
    ho = df[df["date"] >= HOLD_START].dropna(subset=[t])
    sg = ho[ho["date"] >= SURGE_START]
    rep("학습 구간", f"{tr['date'].min()} ~ {tr['date'].max()}", f"{len(tr):,}행")
    rep("홀드아웃", f"{ho['date'].min()} ~ {ho['date'].max()}", f"{len(ho):,}행")
    rep("급등 구간", f"{SURGE_START} 이후", f"{len(sg):,}행")

    print("\n■ ② 학습·평가")
    res = {}
    compare = [("기준", base),
               ("수준변수 투입", list(dict.fromkeys(base + new)))]
    for tag, feats in compare:
        p_all, mdl = fit_predict(tr, ho, feats, t)
        r1 = block("전체", p_all, ho, t)
        idx = ho["date"] >= SURGE_START
        r2 = block("급등", p_all[idx.values], ho[idx], t)
        res[tag] = (r1, r2, mdl, feats)
        print(f"  [{tag}]")
        for r in (r1, r2):
            print(f"    {r['구간']:<5} 표본 {r['표본']:>5}  MAE {r['MAE']:>6.2f}"
                  f"  rMAE {r.get('rMAE', float('nan')):>6.3f}"
                  f"  MAPE {r['MAPE']:>6.2f}%  편의 {r['편의']:>+6.2f}%")
        rep(f"{tag} 전체", f"MAE {r1['MAE']} / rMAE {r1.get('rMAE')}",
            f"MAPE {r1['MAPE']}% / 편의 {r1['편의']}%")
        rep(f"{tag} 급등", f"MAE {r2['MAE']} / rMAE {r2.get('rMAE')}",
            f"MAPE {r2['MAPE']}% / 편의 {r2['편의']}%")

    print("\n■ ③ 변수 중요도 - 신설 변수")
    mdl, feats = res["수준변수 투입"][2], res["수준변수 투입"][3]
    imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
    rank = {c: int(np.where(imp.index == c)[0][0]) + 1 for c in new if c in imp.index}
    for c in new:
        if c in imp.index:
            print(f"    {c:<22} 중요도 {imp[c]:>7.0f}   순위 {rank[c]:>3}/{len(feats)}")
            rep(f"중요도 {c}", f"{imp[c]:.0f}", f"순위 {rank[c]}/{len(feats)}")

    print("\n■ ④ 판정")
    b1, b2 = res["기준"][0], res["기준"][1]
    n1, n2 = res["수준변수 투입"][0], res["수준변수 투입"][1]
    d_all = n1["MAE"] - b1["MAE"]
    d_sg = n2["MAE"] - b2["MAE"]
    print(f"    전체 MAE {b1['MAE']:.2f} -> {n1['MAE']:.2f}  ({d_all:+.2f}원)")
    print(f"    급등 MAE {b2['MAE']:.2f} -> {n2['MAE']:.2f}  ({d_sg:+.2f}원)")
    ok_j = (d_all < 0) and (d_sg < 0)
    verdict = "채택 권고" if ok_j else ("조건부 - 급등 구간만 개선"
                                    if d_sg < 0 else "기각")
    rep("전체 MAE 변화", f"{d_all:+.2f}원", f"{b1['MAE']} -> {n1['MAE']}")
    rep("급등 MAE 변화", f"{d_sg:+.2f}원", f"{b2['MAE']} -> {n2['MAE']}")
    rep("판정", verdict, "전체·급등 동시 개선 시 채택")
    print(f"    판정 : {verdict}")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_level_test_{t}_{ts}.csv")
    pd.DataFrame(_log).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 로그 -> {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
