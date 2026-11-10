# -*- coding: utf-8 -*-
"""
step8_error_decompose.py — 특정일 예측오차 구조 진단 (v1.0)
==============================================================================
배경
  · 2026. 9. 8. 산출에서 주간·야간 과대예측 확인
    - 13시 : 예측 168.81 / 실적 107.58 → **+61.23원**
    - 21~24시 : 실적 대비 과대 경향
  · 9. 7. 진단(주간 10~16시 편의 +10.60%)과 동일 방향·확대
  · γ1.5 적용 후 첫 산출에서 발생 → 운영단 보정만으로 불충분

목적
  · 대상일 24시간 오차를 **보정 전·보정량·잔여오차**로 분해
  · 계통 변수(태양광·수요·기온)와의 상관으로 원인 가설 검증

검증 가설
  ① 태양광 출력 급증(Duck Curve 심화) — 주간 SMP 급락 미추종
  ② 전력수요 감소(냉방수요 소멸)
  ③ 한계발전기 전환 — SMP 수준대 구조 변화
  ④ 편의 보정 방향 오류 — 보정량 부호·크기
  ⑤ 학습 표본의 9월 대표성 부족 — 과거 9월 시간대 곡선 대조

실행
  python scripts\\step8_error_decompose.py --date 2026-09-08
  python scripts\\step8_error_decompose.py --date 2026-09-08 --target smp_jeju

버전 : v1.0 (2026-09-08)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET = os.path.join(BASE_DIR, "data", "processed", "smp_dataset_hourly.csv")

DAY_HOURS = list(range(10, 17))     # 주간
EVE_HOURS = [21, 22, 23, 24]        # 야간 후반
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<22} {str(값):<24} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def load_pred(t, day):
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 예측 이력 없음 → {p}")
    h = pd.read_csv(p, encoding="utf-8-sig")
    h["date"] = h["date"].astype(str)
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    d = h[h["date"] == day].sort_values("hour_n")
    if d.empty:
        sys.exit(f"[중단] {day} 예측 이력 없음")
    return d


def load_ctx(day, t):
    cols = ["date", "hour_n", "smp_land", "smp_jeju", "demand_fcst_land",
            "demand_actual", "pv_land", "pv_total", "temp_land"]
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False)
    d.columns = [c.lstrip("﻿") for c in d.columns]
    keep = [c for c in cols if c in d.columns]
    d = d[keep].copy()
    d["date"] = d["date"].astype(str).str[:10]
    return d


def bias_parts(t, day):
    """산출 시점 보정량 재현 — 운영 로직과 동일"""
    try:
        from step6_bias_adjust import hourly_bias, GAMMA, CAP
        b, note = hourly_bias(t, day)
        if b is None:
            return None, 0.0, 0.0, note
        return b, GAMMA, CAP, note
    except Exception as e:
        return None, 0.0, 0.0, f"재현 실패({type(e).__name__})"


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser(description="특정일 예측오차 구조 진단")
    ap.add_argument("--date", required=True, help="대상일 YYYY-MM-DD")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--years", type=int, default=3, help="과거 9월 대조 연수")
    args = ap.parse_args()
    t, day = args.target, args.date[:10]

    print("=" * 78)
    print(" step8_error_decompose — 예측오차 구조 진단 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 대상 : {day} / {t}")
    print("=" * 78)

    p = load_pred(t, day)
    ctx = load_ctx(day, t)
    cur = ctx[ctx["date"] == day]
    m = p.merge(cur, on=["date", "hour_n"], how="left")
    if m["실적"].isna().all() and t in m.columns:
        m["실적"] = m[t]
    m["오차"] = m["예측"] - m["실적"]

    #  보정량 재현
    b, g, cap, note = bias_parts(t, day)
    if b is not None:
        adj = np.clip(g * m["hour_n"].map(b).fillna(0).values, -cap, cap)
        m["보정량"] = np.round(adj, 2)
        m["보정전"] = np.round(m["예측"] - adj, 2)
        m["보정전오차"] = m["보정전"] - m["실적"]
    else:
        m["보정량"] = np.nan
        m["보정전"] = np.nan
        m["보정전오차"] = np.nan

    print("\n■ ① 24시간 오차 분해")
    print(f"  {'시':>3}{'예측':>9}{'실적':>9}{'오차':>9}"
          f"{'보정량':>8}{'보정전':>9}{'보정전오차':>11}")
    print("  " + "-" * 62)
    for _, r in m.iterrows():
        print(f"  {int(r['hour_n']):>3}{r['예측']:>9.2f}{r['실적']:>9.2f}"
              f"{r['오차']:>9.2f}{r['보정량']:>8.2f}{r['보정전']:>9.2f}"
              f"{r['보정전오차']:>11.2f}")
    print("  " + "-" * 62)
    rep("보정 이력", note, f"γ={g} / 상한 ±{cap:.0f}원")

    print("\n■ ② 구간별 요약")
    def seg(nm, hrs):
        s = m[m["hour_n"].isin(hrs)]
        rep(nm, f"MAE {s['오차'].abs().mean():.2f}원",
            f"평균오차 {s['오차'].mean():+.2f} / 보정전 {s['보정전오차'].mean():+.2f} / "
            f"보정 {s['보정량'].mean():+.2f}")
    seg("전체 24시간", range(1, 25))
    seg("주간 10~16시", DAY_HOURS)
    seg("야간 21~24시", EVE_HOURS)
    seg("심야 1~6시", range(1, 7))

    print("\n■ ③ 계통 변수 상관 — 가설 ①②")
    for c, nm in [("pv_land", "태양광(육지)"), ("pv_total", "태양광(전국)"),
                  ("demand_fcst_land", "수요예측"), ("demand_actual", "수요실적"),
                  ("temp_land", "기온")]:
        if c in m.columns and m[c].notna().any():
            rep(f"오차 ↔ {nm}", f"r = {corr(m['오차'], m[c]):+.3f}",
                f"당일 평균 {m[c].mean():,.1f}")
    print("    ※ 태양광과 **양(+)의 상관**이 크면 가설 ①(Duck Curve 미추종) 지지")

    print("\n■ ④ 최근 추세 — 가설 ③")
    recent = ctx[ctx["date"] < day].tail(24 * 14)
    if not recent.empty and t in recent.columns:
        r7 = recent[recent["date"] >= sorted(recent["date"].unique())[-7]]
        rep("직전 7일 실적 평균", f"{r7[t].mean():.2f}원",
            f"당일 {m['실적'].mean():.2f}원 / 예측 {m['예측'].mean():.2f}원")
        d7 = r7.groupby("hour_n")[t].mean()
        rep("직전 7일 주간 평균", f"{d7.loc[DAY_HOURS].mean():.2f}원",
            f"당일 주간 실적 {m[m['hour_n'].isin(DAY_HOURS)]['실적'].mean():.2f}원")

    print("\n■ ⑤ 과거 9월 시간대 곡선 대조 — 가설 ⑤")
    ctx["_y"] = ctx["date"].str[:4].astype(int)
    ctx["_m"] = ctx["date"].str[5:7].astype(int)
    yr = int(day[:4])
    past = ctx[(ctx["_m"] == 9) & (ctx["_y"] < yr) & (ctx["_y"] >= yr - args.years)]
    if not past.empty and t in past.columns:
        pc = past.groupby("hour_n")[t].mean()
        pk_h = int(pc.idxmax()); tr_h = int(pc.idxmin())
        rep("과거 9월 최고시간", f"{pk_h}시 {pc.max():.1f}원",
            f"{args.years}년 평균 / 최저 {tr_h}시 {pc.min():.1f}원")
        cd = m.set_index("hour_n")["실적"]
        rep("당일 실적 최고시간", f"{int(cd.idxmax())}시 {cd.max():.1f}원",
            f"최저 {int(cd.idxmin())}시 {cd.min():.1f}원")
        cp = m.set_index("hour_n")["예측"]
        rep("당일 예측 최고시간", f"{int(cp.idxmax())}시 {cp.max():.1f}원",
            f"최저 {int(cp.idxmin())}시 {cp.min():.1f}원")
        print("    ※ 실적 곡선의 저점이 주간으로 이동했는데 예측이 심야 저점을 유지하면")
        print("       → **가설 ①(태양광 유입 구조) 지지**")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_error_decompose_{t}_{day}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        m.to_csv(f, index=False)
        f.write("\n")
        pd.DataFrame(_rows).to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
