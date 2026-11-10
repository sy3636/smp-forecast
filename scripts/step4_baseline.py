# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 벤치마크(기준선) 모델
================================================
목적 : 학습형 모델 도입 이전에 단순 규칙 기반 벤치마크의 성능을 확정하여
       이후 모델의 개선율 산정 기준선을 확보
입력 : data/processed/smp_features_hourly.csv (step3_features 산출)
출력 : logs/step4_baseline_summary_{일시}.csv   (벤치마크별 성능)
       logs/step4_baseline_decomp_{일시}.csv    (최우수 벤치마크 오차 분해)
       data/processed/baseline_predictions.csv  (예측값 원본)

벤치마크 정의 (전부 D-1 23시 시점에 산출 가능)
  B1 전일 동시간        : ŷ(t) = y(t-24h)
  B2 전주 동시간        : ŷ(t) = y(t-168h)
  B3 동시간대 7일 평균  : ŷ(t) = mean(y(t-24h), …, y(t-168h)) — 동일 시각
  B4 전일 일평균        : ŷ(t) = mean(전일 24시간)
  B5 동일 일유형 직전   : ŷ(t) = 동일 평일·휴일 구분의 직전 일자 동시간
  B6 B1·B3 평균         : 지속성과 평활의 결합

평가 설계
  · 학습·검증 : 2021-02-01 ~ 2025-12-31, 확장창(expanding window) 5분할
  · 홀드아웃  : 2026-01-01 ~ 2026-07-30 (1회 평가)
  · 지표      : MAPE(%), sMAPE(%), RMSE(원/kWh), MAE(원/kWh), R²

버전 : v1.0 (2026-07-31)
실행 : python scripts\\step4_baseline.py
       python scripts\\step4_baseline.py --target smp_jeju
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
def feat_paths(target):
    """타깃별 피처셋 경로. 없으면 구버전 통합 파일로 대체."""
    a = os.path.join(PROCESSED, f"smp_features_{target}.csv")
    b = os.path.join(PROCESSED, f"feature_dictionary_{target}.csv")
    if os.path.exists(a):
        return a, (b if os.path.exists(b) else
                   os.path.join(PROCESSED, "feature_dictionary.csv"))
    return (os.path.join(PROCESSED, "smp_features_hourly.csv"),
            os.path.join(PROCESSED, "feature_dictionary.csv"))

TRAIN_START = "2021-02-01"
CV_END = "2025-12-31"
HOLD_START = "2026-01-01"
N_FOLD = 5
# ------------------------------------------------------------------
# MAPE 분모 하한 (원/kWh)
#   근거 : 저가 구간(경부하 시간대 태양광 과잉출력에 따른 가격 붕괴)에서
#          MAPE가 발산하여 모델 성능을 과소평가함
#          2026년 홀드아웃 분석 결과, 50원 미만 9행(전체의 0.18%)이
#          전체 MAPE의 약 19%를 유발
#   결정 : 20원 미만 구간은 MAPE 산정에서 제외하고 RMSE·MAE로 평가
#          (2026-08-01 확정, 전력시장 실무상 저가 구간은 절대오차가 중요)
# ------------------------------------------------------------------
MAPE_FLOOR = 20.0
EPS = MAPE_FLOOR   # 하위 호환


def mape_floor():
    """현재 MAPE 분모 하한 반환(모듈 간 동기 참조용)."""
    return MAPE_FLOOR


def set_mape_floor(v):
    """MAPE 분모 하한 변경(전 스크립트 공통 적용)."""
    global MAPE_FLOOR, EPS
    MAPE_FLOOR = EPS = float(v)


# ------------------------------------------------------------------
# 성능 지표
# ------------------------------------------------------------------
def metrics(y, p):
    m = y.notna() & p.notna()
    y, p = y[m], p[m]
    if len(y) == 0:
        return {k: np.nan for k in ("MAPE", "sMAPE", "RMSE", "MAE", "R2", "N")}
    err = p - y
    valid = y.abs() >= MAPE_FLOOR                # MAPE 산정 대상
    mape = (err[valid] / y[valid]).abs().mean() * 100 if valid.any() else np.nan
    smape = (2 * err.abs() / (y.abs() + p.abs()).replace(0, np.nan)).mean() * 100
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"MAPE": mape, "sMAPE": smape, "RMSE": rmse, "MAE": mae,
            "R2": r2, "N": int(len(y))}


def fmt(d):
    return (f"{d['MAPE']:6.2f} {d['sMAPE']:7.2f} {d['RMSE']:8.2f} "
            f"{d['MAE']:7.2f} {d['R2']:6.3f} {d['N']:>7,}")


# ------------------------------------------------------------------
# 벤치마크 생성
# ------------------------------------------------------------------
def build_baselines(df, t):
    """벤치마크별 예측 열 생성. 전부 예측 시점(D-1 23시) 기준 확보 가능."""
    b = pd.DataFrame(index=df.index)
    b["B1_전일동시간"] = df[t].shift(24)
    b["B2_전주동시간"] = df[t].shift(168)

    # B3 : 동일 시각의 직전 7일 평균
    b["B3_동시간7일평균"] = df.groupby("hour_n")[t].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean())

    # B4 : 전일 일평균
    daily = df.groupby("date")[t].mean().rename("dmean")
    dm = daily.shift(1).reset_index().rename(columns={"dmean": "B4_전일일평균"})
    b["B4_전일일평균"] = df[["date"]].merge(dm, on="date", how="left")["B4_전일일평균"].values

    # B5 : 동일 일유형(평일·휴일)의 직전 일자 동시간
    key = df["daytype"].astype(str) + "_" + df["hour_n"].astype(str)
    b["B5_동일일유형직전"] = df.groupby(key)[t].shift(1)

    # B6 : B1·B3 결합
    b["B6_B1B3평균"] = b[["B1_전일동시간", "B3_동시간7일평균"]].mean(axis=1)
    return b


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step4 벤치마크 성능 평가")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--folds", type=int, default=N_FOLD)
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR,
                    help="MAPE 분모 하한(원/kWh). 기본 20")
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    SRC, _ = feat_paths(t)
    if not os.path.exists(SRC):
        sys.exit(f"[중단] 입력파일 없음 → {SRC}\n       step3_features.py 선행 실행 필요")
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 78)
    print(" step4 벤치마크(기준선) 성능 평가 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    b = build_baselines(df, t)
    df = pd.concat([df, b], axis=1)
    names = list(b.columns)

    cv = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].copy()
    ho = df[df["date"] >= HOLD_START].copy()
    print(f"\n■ 평가 구간")
    print(f"  · 교차검증 : {cv['date'].min()} ~ {cv['date'].max()}  {len(cv):,}행")
    print(f"  · 홀드아웃 : {ho['date'].min()} ~ {ho['date'].max()}  {len(ho):,}행")
    n_zero = int((cv[t].abs() < MAPE_FLOOR).sum() + (ho[t].abs() < MAPE_FLOOR).sum())
    if n_zero:
        print(f"  · MAPE 제외 : {n_zero}행 (|SMP| < {MAPE_FLOOR:.0f}원, 저가구간 왜곡 방지)")

    rows = []

    # ── ① 확장창 교차검증 ──────────────────────
    print(f"\n■ ① 확장창 시계열 교차검증 ({args.folds}분할)")
    days = sorted(cv["date"].unique())
    size = len(days) // (args.folds + 1)
    print(f"  {'벤치마크':<16}{'MAPE':>7}{'sMAPE':>8}{'RMSE':>9}{'MAE':>8}{'R²':>7}{'N':>8}")
    print("  " + "-" * 63)
    cv_agg = {n: [] for n in names}
    for k in range(args.folds):
        va_days = set(days[size * (k + 1): size * (k + 2)])
        va = cv[cv["date"].isin(va_days)]
        for n in names:
            cv_agg[n].append(metrics(va[t], va[n]))
    for n in names:
        mm = {k: float(np.nanmean([f[k] for f in cv_agg[n]]))
              for k in ("MAPE", "sMAPE", "RMSE", "MAE", "R2")}
        mm["N"] = int(np.sum([f["N"] for f in cv_agg[n]]))
        print(f"  {n:<16}{fmt(mm)}")
        rows.append({"구간": "교차검증(평균)", "벤치마크": n, **{k: round(v, 3) for k, v in mm.items()}})

    # ── ② 홀드아웃 ─────────────────────────────
    print(f"\n■ ② 홀드아웃 평가 ({HOLD_START} ~ {ho['date'].max()})")
    print(f"  {'벤치마크':<16}{'MAPE':>7}{'sMAPE':>8}{'RMSE':>9}{'MAE':>8}{'R²':>7}{'N':>8}")
    print("  " + "-" * 63)
    ho_res = {}
    for n in names:
        mm = metrics(ho[t], ho[n])
        ho_res[n] = mm
        print(f"  {n:<16}{fmt(mm)}")
        rows.append({"구간": "홀드아웃", "벤치마크": n, **{k: round(v, 3) for k, v in mm.items()}})

    best = min(ho_res, key=lambda k: ho_res[k]["MAPE"])
    print(f"\n  → 최우수 벤치마크 : {best} (홀드아웃 MAPE {ho_res[best]['MAPE']:.2f}%)")
    print(f"  → 이후 모델은 본 수치 대비 개선율로 평가")

    # ── ③ 오차 분해 ────────────────────────────
    print(f"\n■ ③ 오차 분해 — {best} (홀드아웃)")
    ho = ho.copy()
    ho["_err"] = ho[best] - ho[t]
    ho["_ape"] = (ho["_err"] / ho[t].where(ho[t].abs() >= MAPE_FLOOR)).abs() * 100
    dec = []

    def decomp(label, keycol):
        g = ho.groupby(keycol).agg(MAPE=("_ape", "mean"),
                                   RMSE=("_err", lambda x: float(np.sqrt((x ** 2).mean()))),
                                   BIAS=("_err", "mean"), N=("_err", "size"))
        print(f"\n  [{label}]")
        print(f"  {'구분':<10}{'MAPE':>8}{'RMSE':>9}{'편의':>9}{'N':>8}")
        for i, r in g.iterrows():
            print(f"  {str(i):<10}{r['MAPE']:>8.2f}{r['RMSE']:>9.2f}{r['BIAS']:>9.2f}{int(r['N']):>8,}")
            dec.append({"분해기준": label, "구분": str(i), "MAPE": round(r["MAPE"], 2),
                        "RMSE": round(r["RMSE"], 2), "편의": round(r["BIAS"], 2),
                        "N": int(r["N"])})

    decomp("시간대별", "hour_n")
    decomp("월별", "month")
    decomp("일유형별", "daytype")

    # ── ④ 저장 ─────────────────────────────────
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    p1 = os.path.join(LOG_DIR, f"step4_baseline_summary_{t}_{ts}.csv")
    pd.DataFrame(rows).to_csv(p1, index=False, encoding="utf-8-sig")
    p2 = os.path.join(LOG_DIR, f"step4_baseline_decomp_{t}_{ts}.csv")
    pd.DataFrame(dec).to_csv(p2, index=False, encoding="utf-8-sig")
    p3 = os.path.join(PROCESSED, f"baseline_predictions_{t}.csv")
    df[["date", "hour_n", t] + names].to_csv(p3, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f" 성능요약 : {p1}")
    print(f" 오차분해 : {p2}")
    print(f" 예측값   : {p3}")
    print(f" 기준선   : {best} / 홀드아웃 MAPE {ho_res[best]['MAPE']:.2f}% "
          f"/ RMSE {ho_res[best]['RMSE']:.2f}원-kWh")
    print(" ※ Gate 3 기준(MAPE ≤ 8.0%) 대비 위치를 확인하여 모델 고도화 범위 결정")
    print("=" * 78)


if __name__ == "__main__":
    main()
