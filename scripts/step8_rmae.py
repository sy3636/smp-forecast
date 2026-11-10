# -*- coding: utf-8 -*-
"""
step8_rmae.py — 4지표(MAE·rMAE·MAPE·편의) 정례 산출 (v1.0)
==============================================================================
목적
  · 운영 예측 이력을 기준으로 **MAE·rMAE·MAPE·편의** 4지표를 월별 산출
  · 단계 목표 대비 달성 여부를 6개월 이동평균으로 판정

지표 체계 (PRD v2.13)
  | 구분 | 지표 | 산식 | 역할 |
  |------|------|------|------|
  | 주지표① | **MAE** | mean(|예측−실적|) | 실무 체감 오차(원/kWh) |
  | 주지표② | **rMAE** | MAE_모델 ÷ MAE_naive | 모델 기여도 — 국제 준거 |
  | 보조① | MAPE | mean(|오차| ÷ max(실적,20)) × 100 | 대내 보고 연속성 |
  | 보조② | 편의 | mean(오차 ÷ max(실적,20)) × 100 | 방향성 오차 |

naive 정의 (**고정**)
  · **naive-1 : 전일 동시간 실적**(smp_lag_d1) — 판정 기준
  · naive-2 : 동일 일유형 직전 동시간 — 참고 병기
  · rMAE는 naive 정의에 결과가 좌우되므로 **변경 금지**. 변경 시 전 이력 재산출 필요

준거
  · Lago et al., "Forecasting day-ahead electricity prices: A review of
    state-of-the-art algorithms, best practices and an open-access benchmark",
    Applied Energy 293 (2021)
  · 동 문헌은 MAPE 단독 사용을 지양하고 **MAE·rMAE 병용**을 권고함
  · 다만 SOTA 수준(rMAE≈0.4)은 Nord Pool 기준이며, 시장별 naive 강도가
    상이하므로 **직접 비교 시 시장 특성 병기 필요**

단계 목표 (PRD v2.13)
  | 단계 | 시점 | MAE | rMAE | MAPE | 편의 |
  |------|------|-----|------|------|------|
  | 1단계 | 2026.12 | ≤ 8.0원 | ≤ 0.75 | < 6.0% | ≤ 3.0% |
  | 2단계 | 2027.06 | ≤ 7.5원 | ≤ 0.70 | < 5.8% | ≤ 2.0% |
  | 도전  | 2027.12 | ≤ 7.0원 | ≤ 0.65 | < 5.0% | ≤ 1.5% |

실행
  python scripts\\step8_rmae.py                    (육지 운영 이력)
  python scripts\\step8_rmae.py --source holdout   (홀드아웃 기준)
  python scripts\\step8_rmae.py --target smp_jeju

산출
  · data/interim/metrics_monthly_{타깃}.csv   — 대시보드 입력
  · logs/step8_rmae_{일시}.csv

버전 : v1.0 (2026-09-02)
"""
import os
import sys
import argparse
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_baseline import feat_paths  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")

FLOOR = 20.0          # MAPE·편의 분모 하한(원/kWh) — PRD v2.11 승계
NAIVE = "smp_lag_d1"  # **판정 기준 naive — 변경 금지**
NAIVE2 = "smp_lag_same_daytype"

# 단계 목표 — (MAE, rMAE, MAPE, 편의절대값)
TARGET = {
    "1단계(2026.12)": (8.0, 0.75, 6.0, 3.0),
    "2단계(2027.06)": (7.5, 0.70, 5.8, 2.0),
    "도전(2027.12)": (7.0, 0.65, 5.0, 1.5),
}

_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {값:<24} {비고}")
    _rows.append({"항목": 항목, "값": 값, "비고": 비고})


def load_pairs(t, source):
    """예측·실적·naive 결합

    source = operation : 운영 예측 이력(predictions_*.csv) — 판정 기준
             holdout   : 피처셋 전체 구간 — 참고용 비교
    """
    src, _ = feat_paths(t)
    use = ["date", "hour_n", t, NAIVE]
    a = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    if NAIVE2 in a.columns:
        use.append(NAIVE2)
    a = a[[c for c in use if c in a.columns]]

    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 예측 이력 없음 → {p}")
    h = pd.read_csv(p, encoding="utf-8-sig")
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    d = h[["date", "hour_n", "예측"]].merge(a, on=["date", "hour_n"], how="inner")
    return d.dropna(subset=[t, "예측", NAIVE])


def metrics4(y, p, n1, n2=None):
    """MAE·rMAE·MAPE·편의 산출"""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    den = np.maximum(y, FLOOR)
    mae = float(np.abs(p - y).mean())
    mae1 = float(np.abs(np.asarray(n1, float) - y).mean())
    out = {
        "표본": len(y),
        "MAE": round(mae, 2),
        "rMAE": round(mae / mae1, 3) if mae1 > 0 else np.nan,
        "MAPE": round(float((np.abs(p - y) / den).mean() * 100), 2),
        "편의": round(float(((p - y) / den).mean() * 100), 2),
        "naive_MAE": round(mae1, 2),
        "개선율": round((1 - mae / mae1) * 100, 1) if mae1 > 0 else np.nan,
    }
    if n2 is not None and np.isfinite(np.asarray(n2, float)).any():
        mae2 = float(np.abs(np.asarray(n2, float) - y).mean())
        out["rMAE_naive2"] = round(mae / mae2, 3) if mae2 > 0 else np.nan
    return out


def judge(row):
    """단계 목표 대비 판정 — 4지표 전량 충족 시 달성"""
    res = []
    for k, (mae, rmae, mape, bias) in TARGET.items():
        ok = (row["MAE"] <= mae and row["rMAE"] <= rmae
              and row["MAPE"] < mape and abs(row["편의"]) <= bias)
        res.append((k, ok))
    return res


def main():
    ap = argparse.ArgumentParser(description="4지표 정례 산출")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--source", default="operation",
                    choices=["operation", "holdout"])
    args = ap.parse_args()
    t = args.target

    print("=" * 76)
    print(" step8_rmae — 4지표(MAE·rMAE·MAPE·편의) 정례 산출 v1.0")
    print(f" 실행시각 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 76)

    d = load_pairs(t, args.source)
    if d.empty:
        sys.exit("[중단] 대조 가능한 실적 없음")
    d["ym"] = d["date"].str[:7]

    print("\n■ ① 대상")
    rep("구간", f"{d['date'].min()} ~ {d['date'].max()}", f"{len(d):,}행")
    rep("naive 정의", "전일 동시간(smp_lag_d1)", "**판정 기준 — 변경 금지**")
    rep("실적 평균", f"{d[t].mean():.1f}원/kWh", f"분모 하한 {FLOOR:.0f}원")

    n2 = d[NAIVE2] if NAIVE2 in d.columns else None

    print("\n■ ② 전체 구간")
    tot = metrics4(d[t], d["예측"], d[NAIVE], n2)
    print(f"  {'MAE':>8}{'rMAE':>8}{'MAPE':>9}{'편의':>9}{'개선율':>9}")
    print("  " + "-" * 43)
    print(f"  {tot['MAE']:>8.2f}{tot['rMAE']:>8.3f}"
          f"{tot['MAPE']:>8.2f}%{tot['편의']:>8.2f}%{tot['개선율']:>8.1f}%")
    for k, v in tot.items():
        rep(f"전체 {k}", str(v))

    print("\n■ ③ 월별")
    rows = []
    for ym, g in d.groupby("ym"):
        r = metrics4(g[t], g["예측"], g[NAIVE],
                     g[NAIVE2] if NAIVE2 in g.columns else None)
        r["ym"] = ym
        rows.append(r)
    M = pd.DataFrame(rows)[["ym", "표본", "MAE", "rMAE", "MAPE", "편의",
                            "naive_MAE", "개선율"]]
    print(M.to_string(index=False))

    # 6개월 이동평균 — 판정 기준
    if len(M) >= 2:
        print("\n■ ④ 6개월 이동평균 (판정 기준)")
        M = M.sort_values("ym").reset_index(drop=True)
        for c in ("MAE", "rMAE", "MAPE", "편의"):
            M[f"{c}_MA6"] = M[c].rolling(6, min_periods=1).mean().round(3)
        print(M[["ym", "MAE_MA6", "rMAE_MA6", "MAPE_MA6", "편의_MA6"]]
              .to_string(index=False))

    print("\n■ ⑤ 단계 목표 대비 판정 (전체 구간 기준)")
    print(f"  {'단계':<16}{'MAE':>8}{'rMAE':>8}{'MAPE':>8}{'편의':>8}  판정")
    print("  " + "-" * 56)
    for k, (mae, rmae, mape, bias) in TARGET.items():
        ok = (tot["MAE"] <= mae and tot["rMAE"] <= rmae
              and tot["MAPE"] < mape and abs(tot["편의"]) <= bias)
        print(f"  {k:<16}{mae:>8.1f}{rmae:>8.2f}{mape:>7.1f}%{bias:>7.1f}%"
              f"  {'달성' if ok else '미달'}")
        rep(f"판정 {k}", "달성" if ok else "미달",
            f"MAE {tot['MAE']}/{mae} · rMAE {tot['rMAE']}/{rmae}")

    os.makedirs(INTERIM, exist_ok=True)
    op = os.path.join(INTERIM, f"metrics_monthly_{t}.csv")
    M.to_csv(op, index=False, encoding="utf-8-sig")
    rep("산출 파일", os.path.relpath(op, BASE_DIR), f"{len(M)}개월")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_rmae_{t}_{ts}.csv")
    pd.DataFrame(_rows).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")


if __name__ == "__main__":
    main()
