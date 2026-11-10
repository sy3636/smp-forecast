# -*- coding: utf-8 -*-
"""
step8_cap_saturation.py — 편의 보정 상한 포화 점검 (v1.0)
==============================================================================
배경 (2026. 9. 8. A-8 실증 부수 확인)
  · 이상치 내성 실증에서 **전 구성의 최대 보정폭이 상한 25.00원에 도달**
  · γ 1.2 → 1.5 상향(2026. 9. 7.) 이후 포화가 실제 발생 중
  · 포화 시 보정이 절단되어 **잔차 축소가 불완전**해짐 — 상한 재설정 검토 필요

점검 항목
  ① 포화 발생률 — 전체 / 시간대별 / 일자별
  ② 포화 방향 — 상향(+25) · 하향(−25)
  ③ 절단 손실 — 원 보정량과 절단 후의 차이
  ④ 상한 대안 비교 — ±25(현행) · ±30 · ±35 · ±40 · 무제한
  ⑤ γ 대안 병행 — 1.5(현행) · 1.35 · 1.2

방법
  · 운영 이력의 '예측'은 보정 후 값 → 산출 시점 감쇠계수로 보정 전 값 역산
  · 보정계수는 대상일 이전 실적만 사용 — 누출 차단
  · 평일·휴일 2분류 · 창 21일 — 운영과 동일

판정
  · 포화율 10% 이상 : 상한이 실질적 제약 → 상향 검토
  · 상한 상향으로 MAE 개선 시 채택. 단 2026. 9. 6. ±35·±50 기각 이력 대조 필수

실행
  python scripts\\step8_cap_saturation.py
  python scripts\\step8_cap_saturation.py --target smp_jeju

버전 : v1.0 (2026-09-08)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET = os.path.join(BASE_DIR, "data", "processed", "smp_dataset_hourly.csv")

MIN_DAYS, MIN_DAYS_SPLIT = 7, 4
GAMMA_STEPS = [("2026-09-07", 1.5), ("2026-09-05", 1.2), ("0000-00-00", 0.7)]
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<24} {str(값):<22} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def gamma_of(day):
    for d0, g in GAMMA_STEPS:
        if str(day) >= d0:
            return g
    return 0.7


def load_cal():
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c.lstrip("﻿") in ("date", "daytype"))
    d.columns = [c.lstrip("﻿") for c in d.columns]
    d = d.drop_duplicates("date")
    d["date"] = d["date"].astype(str).str[:10]
    d["구분2"] = np.where(d["daytype"].astype(str) == "휴일", "휴일", "평일")
    return d[["date", "구분2"]]


def load_hist(t, cal):
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 운영 이력 없음 → {p}")
    h = pd.read_csv(p, encoding="utf-8-sig").dropna(subset=["예측", "실적"])
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    h["date"] = h["date"].astype(str).str[:10]
    return (h.merge(cal, on="date", how="left")
            .sort_values(["date", "hour_n"]).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description="편의 보정 상한 포화 점검")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--cap", type=float, default=25.0, help="현행 상한(원)")
    ap.add_argument("--cap-list", default="25,30,35,40,999",
                    help="상한 대안(쉼표 구분, 999=무제한)")
    ap.add_argument("--gamma-list", default="1.5,1.35,1.2",
                    help="감쇠 대안(쉼표 구분)")
    args = ap.parse_args()
    t, K = args.target, args.k

    print("=" * 78)
    print(" step8_cap_saturation — 편의 보정 상한 포화 점검 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    cal = load_cal()
    h = load_hist(t, cal)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")
    rep("점검 조건", f"창 {K}일 / 현행 상한 ±{args.cap:.0f}원", "평일·휴일 2분류")

    #  보정 전 역산
    rec = []
    h["보정전"] = h["예측"]
    for d in days:
        seg = h[h["date"] == d].copy()
        base = pd.concat(rec, ignore_index=True) if rec else None
        ok = False
        if base is not None and base["date"].nunique() >= MIN_DAYS:
            pr = base.copy()
            pr["잔차"] = pr["실적"] - pr["보정전"]
            sub = pr[pr["구분2"] == seg["구분2"].iloc[0]]
            use = sub if sub["date"].nunique() >= MIN_DAYS_SPLIT else pr
            sel = sorted(use["date"].unique())[-K:]
            b = use[use["date"].isin(sel)].groupby("hour_n")["잔차"].mean()
            raw = gamma_of(d) * seg["hour_n"].map(b).fillna(0).values
            seg["보정전"] = seg["예측"].values - np.clip(raw, -args.cap, args.cap)
            ok = True
        seg["_적용"] = ok
        rec.append(seg)
    hh = pd.concat(rec, ignore_index=True)
    hh["잔차"] = hh["실적"] - hh["보정전"]
    ev = hh[hh["_적용"]].copy()
    if ev.empty:
        sys.exit("[중단] 평가 표본 없음")
    rep("평가 표본", f"{len(ev):,}행", f"{ev['date'].nunique()}일")

    #  원 보정량(절단 전) 산출
    def raw_adj(g):
        out = []
        for d in sorted(ev["date"].unique()):
            seg = ev[ev["date"] == d]
            pr = hh[hh["date"] < d]
            sub = pr[pr["구분2"] == seg["구분2"].iloc[0]]
            use = sub if sub["date"].nunique() >= MIN_DAYS_SPLIT else pr
            sel = sorted(use["date"].unique())[-K:]
            s = use[use["date"].isin(sel)]
            b = (s.groupby("hour_n")["잔차"].mean() if not s.empty
                 else pd.Series(dtype=float))
            out.append(pd.DataFrame({
                "date": seg["date"].values, "hour_n": seg["hour_n"].values,
                "실적": seg["실적"].values, "보정전": seg["보정전"].values,
                "원보정": g * seg["hour_n"].map(b).fillna(0).values}))
        return pd.concat(out, ignore_index=True)

    gammas = [float(x) for x in args.gamma_list.split(",") if x.strip()]
    caps = [float(x) for x in args.cap_list.split(",") if x.strip()]
    G0 = gammas[0]
    r0 = raw_adj(G0)
    sat = r0["원보정"].abs() > args.cap
    print(f"\n■ ② 포화 현황 (γ {G0} · 상한 ±{args.cap:.0f}원)")
    rep("포화 행수", f"{int(sat.sum()):,}행 / {len(r0):,}행",
        f"포화율 **{sat.mean() * 100:.1f}%**")
    rep("포화 방향", f"상향 {int((r0['원보정'] > args.cap).sum())}행 / "
        f"하향 {int((r0['원보정'] < -args.cap).sum())}행", "")
    rep("원 보정량 범위", f"{r0['원보정'].min():+.1f} ~ {r0['원보정'].max():+.1f}원",
        f"절대 최대 {r0['원보정'].abs().max():.1f}원")
    if sat.any():
        loss = (r0.loc[sat, "원보정"].abs() - args.cap)
        rep("절단 손실", f"평균 {loss.mean():.1f}원 / 최대 {loss.max():.1f}원",
            "상한 초과분")
        bh = r0[sat].groupby("hour_n").size().sort_values(ascending=False)
        rep("포화 집중 시간대", ", ".join(f"{int(i)}시({v})" for i, v in bh.head(5).items()),
            "상위 5")
        bd = r0[sat].groupby("date").size().sort_values(ascending=False)
        rep("포화 집중 일자", ", ".join(f"{i}({v})" for i, v in bd.head(5).items()),
            "상위 5")

    print("\n■ ③ 상한·감쇠 격자 — MAE(원)")
    print(f"  {'γ':>6}" + "".join(f"{('무제한' if c > 100 else f'±{c:.0f}'):>10}"
                                  for c in caps))
    print("  " + "-" * (6 + 10 * len(caps)))
    grid = {}
    for g in gammas:
        rg = raw_adj(g)
        row = []
        for c in caps:
            adj = np.clip(rg["원보정"].values, -c, c)
            m = mae(rg["보정전"].values + adj, rg["실적"].values)
            grid[(g, c)] = m
            row.append(m)
        print(f"  {g:>6.2f}" + "".join(f"{v:>10.2f}" for v in row))
    print("  " + "-" * (6 + 10 * len(caps)))
    cur = grid[(G0, args.cap)]
    rep("현행 구성", f"γ {G0} / ±{args.cap:.0f}원 → MAE {cur:.2f}원", "기준")
    best = min(grid, key=grid.get)
    rep("격자 최우수", f"γ {best[0]} / "
        f"{'무제한' if best[1] > 100 else f'±{best[1]:.0f}원'} → MAE {grid[best]:.2f}원",
        f"현행 대비 {grid[best] - cur:+.2f}원")

    print("\n■ ④ 판정")
    print(f"    · 포화율 {sat.mean() * 100:.1f}% — "
          f"{'상한이 실질 제약(10% 이상)' if sat.mean() >= 0.10 else '제약 경미(10% 미만)'}")
    print("    · 2026. 9. 6. ±35·±50 상향 실증은 **기각**(전체 +0.04원 악화) — 대조 필수")
    print("    · 채택 기준 : 현행 대비 MAE 개선 + 과보정 억제 기능 유지")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_cap_saturation_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        pd.DataFrame([{"gamma": k[0], "cap": k[1], "MAE": v}
                      for k, v in grid.items()]).to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
