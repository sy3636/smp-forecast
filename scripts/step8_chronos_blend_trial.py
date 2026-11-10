# -*- coding: utf-8 -*-
"""
step8_chronos_blend_trial.py — Chronos 혼성 가중 시험 점검 (v1.0)
==============================================================================
시험 구간 (2026.9.12 확정)
  · 대상일 : 2026-09-14 ~ 2026-09-27
  · 가중   : w=0.30 (현행 70% + Chronos-cov 30%)
  · 설정   : config/chronos_trial.env → docker-compose 환경변수

판정
  · 채택 후보 : 시험일 중 실적 확보분의 혼성 MAE < 현행 MAE
  · 조기중단 : 혼성 MAE > 현행×1.10 이 3일 연속
  · 2026.9.27 채택 : w=0.30 유지(상향·폐지 아님). 본 스크립트는 사후 점검용

실행
  python scripts\\step8_chronos_blend_trial.py
  python scripts\\step8_chronos_blend_trial.py --w 0.3 --start 2026-09-14 --end 2026-09-27

버전 : v1.0 (2026-09-12)
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FLOOR = 20.0

TRIAL_START = "2026-09-14"
TRIAL_END = "2026-09-27"
TRIAL_W = 0.30


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def bias(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    if not m.any():
        return np.nan
    return float(((p[m] - a[m]) / np.maximum(a[m], FLOOR)).mean() * 100)


def load_ops():
    p = os.path.join(OUT_DIR, "predictions_smp_land.csv")
    h = pd.read_csv(p, encoding="utf-8-sig")
    h["date"] = h["date"].astype(str).str[:10]
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            ["date", "hour_n"], keep="last")
    return h.dropna(subset=["예측", "실적"])


def load_shadow():
    p = os.path.join(OUT_DIR, "chronos_shadow_smp_land.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    h = pd.read_csv(p, encoding="utf-8-sig")
    h["date"] = h["date"].astype(str).str[:10]
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            ["date", "hour_n"], keep="last")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=TRIAL_START)
    ap.add_argument("--end", default=TRIAL_END)
    ap.add_argument("--w", type=float, default=TRIAL_W)
    args = ap.parse_args()

    print("=" * 72)
    print(" Chronos 혼성 가중 시험 점검")
    print(f" 실행 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 구간 : {args.start} ~ {args.end}  /  w={args.w}")
    print("=" * 72)

    ops = load_ops()
    sh = load_shadow()
    if sh.empty:
        print("  · chronos_shadow 없음 — step8_chronos_shadow 선행 필요")
        return

    m = ops.merge(
        sh[["date", "hour_n", "예측"]].rename(columns={"예측": "예측_ch"}),
        on=["date", "hour_n"], how="inner")
    m = m[(m["date"] >= args.start) & (m["date"] <= args.end)]
    if m.empty:
        print("  · 시험 구간 중첩 실적 없음 (시험 진행 중이면 정상)")
        print(f"  · shadow 보유일: {', '.join(sorted(sh['date'].unique())[-5:])}")
        return

    rows = []
    worse = 0
    for day, g in m.groupby("date"):
        blend = (1 - args.w) * g["예측"] + args.w * g["예측_ch"]
        mo, mb = mae(g["예측"], g["실적"]), mae(blend, g["실적"])
        flag = "열위" if mb > mo * 1.10 else ("우위" if mb < mo else "동등")
        if flag == "열위":
            worse += 1
        else:
            worse = 0
        rows.append({
            "date": day, "N": len(g),
            "MAE_ops": round(mo, 2), "MAE_blend": round(mb, 2),
            "Δ": round(mb - mo, 2),
            "편의_ops": round(bias(g["예측"], g["실적"]), 2),
            "편의_blend": round(bias(blend, g["실적"]), 2),
            "판정": flag, "연속열위": worse,
        })
        if worse >= 3:
            print(f"  !! 조기중단 조건 충족 — {day}까지 열위 3일 연속")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n  · 종합 MAE  현행 {mae(m['예측'], m['실적']):.2f}  "
          f"/ 혼성 {mae((1-args.w)*m['예측']+args.w*m['예측_ch'], m['실적']):.2f}  "
          f"(N={len(m)}, {m['date'].nunique()}일)")
    win = (df["판정"] == "우위").sum()
    print(f"  · 일별 우위 {win}/{len(df)}일")
    if df["MAE_blend"].mean() < df["MAE_ops"].mean():
        print("  · 판정: 채택 후보 — 시험 종료 후 w 유지·상향 검토")
    else:
        print("  · 판정: 현행 유리 — 시험 종료 후 w=0 복귀 권고")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_chronos_blend_trial_{ts}.csv")
    df.to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"  · 로그 → {os.path.relpath(lp, BASE_DIR)}")


if __name__ == "__main__":
    main()
