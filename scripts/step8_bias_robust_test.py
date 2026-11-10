# -*- coding: utf-8 -*-
"""
step8_bias_robust_test.py — 편의 보정 이상치 내성 실증 (v1.0)
==============================================================================
배경 (A-8, 2026. 9. 8.)
  · 2026. 9. 8. 산출에서 예외적 대형 오차 발생
    | 시간 | 예측 | 실적 | 오차 |
    |------|-----:|-----:|-----:|
    | 13시 | 168.81 | 107.58 | **+61.23** |
    | 21시 | 182.84 | 137.08 | **+45.76** |
    | 22시 | 174.13 | 125.25 | **+48.88** |
  · A-7 진단 : 야간 급락은 구조적 악화가 아닌 **이례적 단발 사건**(발생빈도 7.2%)
  · 위험 : 해당 오차가 향후 21일간 잔차 창에 유입되어 **정상일 예측을 왜곡**
    — 현행 보정은 시간대별 **평균 잔차**이므로 단일 이상치에 취약
    — γ1.5 적용 시 이론상 최대 ±25원(상한)까지 과대 보정 가능

가설
  · 잔차 산정을 이상치에 강건한 방식으로 전환하면 정상일 성능이 개선됨

비교 구성 (창 21일 · γ1.5 · 상한 ±25원 · 평일·휴일 2분류 — 현행 운영 조건)
  | No | 구성 | 잔차 산정 |
  |----|------|-----------|
  | ①  | 무보정 | — |
  | ②  | **현행(평균)** | 시간대별 산술평균 |
  | ③  | 중위수 | 시간대별 중위수 |
  | ④  | 절사평균 | 상·하위 10% 절사 후 평균 |
  | ⑤  | 이상일 제외 | 일평균 절대오차 > 임계(기본 25원) 일자 제외 후 평균 |
  | ⑥  | 잔차 윈저화 | 시간별 잔차를 ±임계(기본 40원)로 절단 후 평균 |

방법
  · 운영 이력의 '예측'은 보정 후 값 → 산출 시점 감쇠계수로 보정 전 값 역산
    (2026-09-05 이전 γ=0.7 / 09-05~09-06 γ=1.2 / 09-07 이후 γ=1.5)
  · 보정계수는 대상일 이전 실적만 사용 — 누출 차단
  · 구분별 이력 4일 미만 시 전체 잔차로 대체(운영 로직과 동일)

판정
  · 채택 기준 : **전체 MAE 개선 + 정상일 MAE 비악화**
    (정상일 = 일평균 절대오차가 임계 이하인 일자 — 이상일 자체의 오차는 보정 대상 아님)

실행
  python scripts\\step8_bias_robust_test.py
  python scripts\\step8_bias_robust_test.py --target smp_jeju --outlier-th 30

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
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATASET = os.path.join(PROCESSED, "smp_dataset_hourly.csv")

CAP = 25.0
MIN_DAYS = 7
MIN_DAYS_SPLIT = 4
GAMMA_STEPS = [("2026-09-07", 1.5), ("2026-09-05", 1.2), ("0000-00-00", 0.7)]
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<24} {str(값):<22} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def bias_pct(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(((p[m] - a[m]) / np.maximum(a[m], 20.0)).mean() * 100)


def gamma_of(day):
    for d0, g in GAMMA_STEPS:
        if str(day) >= d0:
            return g
    return 0.7


def load_calendar():
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


#  ── 잔차 산정 방식 6종 ────────────────────────────────────────────
def resid_mean(s):
    return s.groupby("hour_n")["잔차"].mean()


def resid_median(s):
    return s.groupby("hour_n")["잔차"].median()


def resid_trimmed(s, q=0.10):
    def _t(x):
        if len(x) < 5:
            return x.mean()
        lo, hi = x.quantile(q), x.quantile(1 - q)
        y = x[(x >= lo) & (x <= hi)]
        return y.mean() if len(y) else x.mean()
    return s.groupby("hour_n")["잔차"].apply(_t)


def resid_exclude(s, th):
    """일평균 절대오차가 임계 초과인 일자를 제외 후 평균"""
    dm = s.groupby("date")["잔차"].apply(lambda x: x.abs().mean())
    keep = dm[dm <= th].index
    ss = s[s["date"].isin(keep)]
    if ss["date"].nunique() < 3:
        ss = s
    return ss.groupby("hour_n")["잔차"].mean()


def resid_winsor(s, th):
    """시간별 잔차를 ±임계로 절단 후 평균"""
    x = s.copy()
    x["잔차"] = x["잔차"].clip(-th, th)
    return x.groupby("hour_n")["잔차"].mean()


def main():
    ap = argparse.ArgumentParser(description="편의 보정 이상치 내성 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--k", type=int, default=21, help="잔차 산정 창(일)")
    ap.add_argument("--gamma", type=float, default=1.5, help="감쇠계수")
    ap.add_argument("--cap", type=float, default=CAP, help="보정 상한(원)")
    ap.add_argument("--outlier-th", type=float, default=25.0,
                    help="이상일 판정 임계 — 일평균 절대오차(원)")
    ap.add_argument("--winsor-th", type=float, default=40.0,
                    help="잔차 윈저화 임계(원)")
    args = ap.parse_args()
    t, K, G, CP = args.target, args.k, args.gamma, args.cap

    print("=" * 78)
    print(" step8_bias_robust_test — 편의 보정 이상치 내성 실증 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    cal = load_calendar()
    h = load_hist(t, cal)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")
    rep("보정 조건", f"창 {K}일 / γ {G} / 상한 ±{CP:.0f}원", "평일·휴일 2분류")
    rep("임계", f"이상일 {args.outlier_th:.0f}원 / 윈저 {args.winsor_th:.0f}원",
        "일평균 절대오차 / 시간별 잔차")

    #  보정 전 역산 — 순차 갱신
    rec, base = [], None
    h["보정전"] = h["예측"]
    for d in days:
        seg = h[h["date"] == d].copy()
        base = pd.concat(rec, ignore_index=True) if rec else None
        ok = False
        if base is not None and base["date"].nunique() >= MIN_DAYS:
            prior = base.copy()
            prior["잔차"] = prior["실적"] - prior["보정전"]
            sub = prior[prior["구분2"] == seg["구분2"].iloc[0]]
            use = sub if sub["date"].nunique() >= MIN_DAYS_SPLIT else prior
            sel = sorted(use["date"].unique())[-K:]
            b = resid_mean(use[use["date"].isin(sel)])
            adj = np.clip(gamma_of(d) * seg["hour_n"].map(b).fillna(0).values,
                          -CP, CP)
            seg["보정전"] = seg["예측"].values - adj
            ok = True
        seg["_적용"] = ok
        rec.append(seg)
    hh = pd.concat(rec, ignore_index=True)
    hh["잔차"] = hh["실적"] - hh["보정전"]
    ev = hh[hh["_적용"]].copy()
    if ev.empty:
        sys.exit("[중단] 평가 표본 없음")

    #  이상일 식별 — 보정 전 기준
    dm = ev.groupby("date")["잔차"].apply(lambda x: x.abs().mean())
    bad = set(dm[dm > args.outlier_th].index)
    rep("평가 표본", f"{len(ev):,}행", f"{ev['date'].nunique()}일")
    rep("이상일", f"{len(bad)}일",
        ", ".join(sorted(bad)) if bad and len(bad) <= 6 else f"임계 {args.outlier_th:.0f}원 초과")

    #  구성별 적용
    def apply_scheme(fn):
        out = []
        for d in sorted(ev["date"].unique()):
            seg = ev[ev["date"] == d]
            prior = hh[hh["date"] < d]
            sub = prior[prior["구분2"] == seg["구분2"].iloc[0]]
            use = sub if sub["date"].nunique() >= MIN_DAYS_SPLIT else prior
            sel = sorted(use["date"].unique())[-K:]
            s = use[use["date"].isin(sel)]
            if s.empty or fn is None:
                adj = np.zeros(len(seg))
            else:
                b = fn(s)
                adj = np.clip(G * seg["hour_n"].map(b).fillna(0).values, -CP, CP)
            out.append(pd.DataFrame({
                "date": seg["date"].values, "hour_n": seg["hour_n"].values,
                "실적": seg["실적"].values,
                "보정후": seg["보정전"].values + adj,
                "보정량": adj}))
        return pd.concat(out, ignore_index=True)

    schemes = [
        ("① 무보정", None),
        ("② 현행 평균", resid_mean),
        ("③ 중위수", resid_median),
        ("④ 절사평균 10%", resid_trimmed),
        ("⑤ 이상일 제외", lambda s: resid_exclude(s, args.outlier_th)),
        ("⑥ 잔차 윈저화", lambda s: resid_winsor(s, args.winsor_th)),
    ]

    print("\n■ ② 구성별 성능")
    print(f"  {'구성':<18}{'전체MAE':>9}{'정상일MAE':>11}{'이상일MAE':>11}"
          f"{'편의%':>9}{'최대보정':>10}")
    print("  " + "-" * 70)
    res = {}
    for nm, fn in schemes:
        r = apply_scheme(fn)
        nor = r[~r["date"].isin(bad)]
        abn = r[r["date"].isin(bad)]
        row = {"전체MAE": mae(r["보정후"], r["실적"]),
               "정상일MAE": mae(nor["보정후"], nor["실적"]),
               "이상일MAE": mae(abn["보정후"], abn["실적"]) if len(abn) else np.nan,
               "편의": bias_pct(r["보정후"], r["실적"]),
               "최대보정": float(np.abs(r["보정량"]).max())}
        res[nm] = row
        print(f"  {nm:<18}{row['전체MAE']:>9.2f}{row['정상일MAE']:>11.2f}"
              f"{row['이상일MAE']:>11.2f}{row['편의']:>9.2f}{row['최대보정']:>10.2f}")
        rep(nm, f"MAE {row['전체MAE']:.2f}",
            f"정상일 {row['정상일MAE']:.2f} / 이상일 {row['이상일MAE']:.2f} / "
            f"편의 {row['편의']:+.2f}%")
    print("  " + "-" * 70)

    cur = res["② 현행 평균"]
    print("\n■ ③ 판정 — 현행 대비")
    for nm, row in res.items():
        if nm.startswith("②") or nm.startswith("①"):
            continue
        print(f"    {nm:<18} 전체 {row['전체MAE'] - cur['전체MAE']:+.2f}원 / "
              f"정상일 {row['정상일MAE'] - cur['정상일MAE']:+.2f} / "
              f"편의 {row['편의'] - cur['편의']:+.2f}%p")
        rep(f"현행 대비 {nm}", f"전체 {row['전체MAE'] - cur['전체MAE']:+.2f}원",
            f"정상일 {row['정상일MAE'] - cur['정상일MAE']:+.2f}")
    best = min((k for k in res if not k.startswith("①")),
               key=lambda k: res[k]["정상일MAE"])
    print(f"\n    최우수(정상일 MAE) : {best}")
    rep("최우수 구성", best, "정상일 MAE 기준")
    print("    ※ 채택 기준 : 전체 MAE 개선 + 정상일 MAE 비악화")
    print("    ※ 이상일 자체의 오차는 보정 대상이 아님 — 정상일 보호가 목적")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_bias_robust_test_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        pd.DataFrame(res).T.reset_index().to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
