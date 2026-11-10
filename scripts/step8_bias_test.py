# -*- coding: utf-8 -*-
"""
step8_bias_test.py — 과소예측 편의·진폭 축소 보정 실증 (v1.0)
==============================================================================
목적
  · 2026.9.1~9.2 산출에서 관측된 **주간·저녁 시간대 계통적 과소예측**의
    원인을 규명하고, 보정 대안별 개선 효과를 홀드아웃에서 정량 비교

진단 결과 (2026.9.2, outputs/predict/predictions_smp_land.csv 17일분)
  ① 수준 편의 : 일평균 수준 ↔ 편의 상관 **−0.896**
     → 수준이 높은 날 과소예측, 낮은 날 과대예측 (평균회귀 축소)
  ② 진폭 축소 : 일중 진폭이 실적의 **88.8%**에 그침
     → 첨두·경부하 양단이 평탄화됨

보정 대안
  | 코드 | 방식 | 산식 |
  |------|------|------|
  | 기준 | 무보정 | ŷ |
  | A    | 시간대별 잔차 보정 | ŷ + mean(실적−예측)_{h, 최근 K일} |
  | B    | 일수준 선형 재보정 | a + b·ŷ (최근 K일 회귀, 일평균 기준) |
  | C    | 진폭 확대 | ȳ_d + λ·(ŷ − ȳ_d), ȳ_d = 당일 예측 평균 |
  | D    | A + C 결합 | 잔차 보정 후 진폭 확대 |

시점 정합성
  · 보정계수는 **대상일 직전까지의 실적**만 사용(walk-forward)
  · 운영 시각(23:30) 기준 D+0 실적은 확보되므로 전량 재현 가능

실행
  python scripts\\step8_bias_test.py
  python scripts\\step8_bias_test.py --target smp_jeju --k 14

산출
  · logs/step8_bias_test_{일시}.csv

버전 : v1.0 (2026-09-02)
"""
import os
import sys
import argparse
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_baseline import (  # noqa: E402
    feat_paths, metrics, TRAIN_START, CV_END, HOLD_START, set_mape_floor,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

CATEGORICAL = ["daytype", "season"]
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju",
           "smp_land_raw", "smp_jeju_raw"}
# 급등 구간 — 보정 효과의 실전 검증 구간
SURGE_START = "2026-08-01"

_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<28} {값:<22} {비고}")
    _rows.append({"항목": 항목, "값": 값, "비고": 비고})


def hour_bias(hist, k):
    """직전 k일 시간대별 평균 잔차(실적−예측) 산출"""
    if hist.empty:
        return {}
    days = sorted(hist["date"].unique())[-k:]
    h = hist[hist["date"].isin(days)]
    return (h["y"] - h["p"]).groupby(h["hour_n"]).mean().to_dict()


def level_coef(hist, k):
    """직전 k일 일평균 기준 선형 재보정 계수 (a, b)"""
    if hist.empty:
        return 0.0, 1.0
    days = sorted(hist["date"].unique())[-k:]
    g = hist[hist["date"].isin(days)].groupby("date").agg(
        y=("y", "mean"), p=("p", "mean"))
    if len(g) < 5 or g["p"].std() < 1e-6:
        return 0.0, 1.0
    b, a = np.polyfit(g["p"], g["y"], 1)
    # 과도 보정 방지 — 기울기 0.7~1.5 제한
    b = float(np.clip(b, 0.7, 1.5))
    return float(a), b


def amp_coef(hist, k):
    """직전 k일 일중 진폭비 λ = 실적 진폭 / 예측 진폭"""
    if hist.empty:
        return 1.0
    days = sorted(hist["date"].unique())[-k:]
    g = hist[hist["date"].isin(days)].groupby("date").agg(
        ay=("y", lambda s: s.max() - s.min()),
        ap=("p", lambda s: s.max() - s.min()))
    if len(g) < 5 or g["ap"].mean() < 1e-6:
        return 1.0
    return float(np.clip(g["ay"].mean() / g["ap"].mean(), 0.9, 1.4))


def evaluate(d, col, label, tag=""):
    """구간별 지표 산출"""
    m = metrics(d["y"], d[col])
    bias = float((d[col] - d["y"]).mean())
    return {"보정": label, "구간": tag, "MAPE": round(m["MAPE"], 3),
            "RMSE": round(m["RMSE"], 3), "MAE": round(m["MAE"], 3),
            "편의": round(bias, 2), "표본": len(d)}


def main():
    ap = argparse.ArgumentParser(description="편의·진폭 보정 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--k", type=int, default=14, help="보정 산정 창(일). 기본 14")
    ap.add_argument("--mape-floor", type=float, default=20.0)
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    print("=" * 78)
    print(" step8_bias_test — 과소예측 편의·진폭 보정 실증 v1.0")
    print(f" 실행시각 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    SRC, DICT = feat_paths(t)
    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")

    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    cats = [c for c in CATEGORICAL if c in feats]
    feats = [c for c in feats
             if c in cats or pd.api.types.is_numeric_dtype(df[c])]
    for c in cats:
        df[c] = df[c].astype("category")

    print("\n■ ① 구성")
    rep("투입 변수", f"{len(feats)}개", f"범주 {len(cats)}개")

    tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].dropna(subset=[t])
    ho = df[df["date"] >= HOLD_START].dropna(subset=[t]).copy()
    rep("학습 구간", f"{tr['date'].min()} ~ {tr['date'].max()}", f"{len(tr):,}행")
    rep("홀드아웃", f"{ho['date'].min()} ~ {ho['date'].max()}", f"{len(ho):,}행")

    print("\n■ ② 학습")
    try:
        from lightgbm import LGBMRegressor as M
        mdl = M(n_estimators=1200, learning_rate=0.05, num_leaves=63,
                min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor as M
        mdl = M(max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
                min_samples_leaf=30, l2_regularization=1.0, random_state=42,
                categorical_features=[feats.index(c) for c in cats])
    mdl.fit(tr[feats], tr[t])
    ho["p"] = mdl.predict(ho[feats])
    ho["y"] = ho[t].values
    rep("학습 완료", f"{type(mdl).__name__}", "홀드아웃 예측 산출")

    # ── ③ walk-forward 보정 ────────────────────
    print("\n■ ③ 보정 산출 (walk-forward, 창 %d일)" % args.k)
    k = args.k
    base = ho[["date", "hour_n", "y", "p"]].copy()
    for c in ("A", "B", "C", "D"):
        base[c] = np.nan

    days = sorted(base["date"].unique())
    for i, dt in enumerate(days):
        cur = base["date"] == dt
        hist = base[base["date"] < dt][["date", "hour_n", "y", "p"]]
        hb = hour_bias(hist, k)
        a, b = level_coef(hist, k)
        lam = amp_coef(hist, k)

        p = base.loc[cur, "p"]
        # A. 시간대별 잔차 보정
        adj = base.loc[cur, "hour_n"].map(hb).fillna(0.0)
        pa = p + adj
        base.loc[cur, "A"] = pa
        # B. 일수준 선형 재보정
        base.loc[cur, "B"] = a + b * p
        # C. 진폭 확대 (당일 예측 평균 중심)
        mu = p.mean()
        base.loc[cur, "C"] = mu + lam * (p - mu)
        # D. A + C 결합
        mua = pa.mean()
        base.loc[cur, "D"] = mua + lam * (pa - mua)

    rep("보정 대상일", f"{len(days)}일", f"초기 {k}일은 보정계수 미성숙")

    # ── ④ 성능 비교 ────────────────────────────
    print("\n■ ④ 성능 비교")
    warm = base[base["date"] >= sorted(days)[min(k, len(days) - 1)]].copy()
    surge = warm[warm["date"] >= SURGE_START]

    LAB = {"p": "기준(무보정)", "A": "A 잔차 보정",
           "B": "B 선형 재보정", "C": "C 진폭 확대", "D": "D 결합(A+C)"}
    res = []
    for c, lab in LAB.items():
        res.append(evaluate(warm, c, lab, "전체"))
    for c, lab in LAB.items():
        if len(surge):
            res.append(evaluate(surge, c, lab, "급등(8월~)"))
    R = pd.DataFrame(res)

    for tag in R["구간"].unique():
        sub = R[R["구간"] == tag]
        b0 = float(sub[sub["보정"] == "기준(무보정)"]["MAPE"].iloc[0])
        print(f"\n  [{tag}]  표본 {int(sub['표본'].iloc[0]):,}행")
        print(f"  {'보정':<16}{'MAPE':>8}{'증감':>9}{'RMSE':>9}{'MAE':>8}{'편의':>9}")
        print("  " + "-" * 60)
        for _, r in sub.iterrows():
            dlt = r["MAPE"] - b0
            mark = "" if r["보정"].startswith("기준") else f"{dlt:+.2f}%p"
            print(f"  {r['보정']:<16}{r['MAPE']:>8.2f}{mark:>9}"
                  f"{r['RMSE']:>9.2f}{r['MAE']:>8.2f}{r['편의']:>9.2f}")
            rep(f"[{tag}] {r['보정']}", f"MAPE {r['MAPE']:.2f}%",
                f"RMSE {r['RMSE']:.2f} / 편의 {r['편의']:+.2f}")

    # ── ⑤ 판정 ─────────────────────────────────
    print("\n■ ⑤ 판정")
    for tag in R["구간"].unique():
        sub = R[(R["구간"] == tag) & (R["보정"] != "기준(무보정)")]
        b0 = float(R[(R["구간"] == tag) &
                     (R["보정"] == "기준(무보정)")]["MAPE"].iloc[0])
        best = sub.loc[sub["MAPE"].idxmin()]
        rep(f"[{tag}] 최우수", best["보정"],
            f"{b0:.2f}% → {best['MAPE']:.2f}% ({best['MAPE'] - b0:+.2f}%p)")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(LOG_DIR, f"step8_bias_test_{t}_{ts}.csv")
    pd.DataFrame(_rows).to_csv(p, index=False, encoding="utf-8-sig")
    R.to_csv(p.replace(".csv", "_matrix.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[완료] 로그 → {os.path.relpath(p, BASE_DIR)}")


if __name__ == "__main__":
    main()
