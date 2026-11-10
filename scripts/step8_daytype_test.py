# -*- coding: utf-8 -*-
"""
step8_daytype_test.py — 요일·휴일 특성 반영 실증 (v1.0)
==============================================================================
배경
  · 토·일 및 공휴일(설·추석 포함)은 전력수요 감소로 SMP 수준이 하락
    - 학습 구간(2021-01-01~2026-09-04, 2072일) 일평균 SMP(육지)
      평일 140.16원 / 휴일 124.31원 — **−11.3%**
      요일별 : 월 134.97 · 화 139.54 · 수 139.82 · 목 140.25 · 금 139.95
               토 130.42 · **일 120.07**
  · 예측 정확도가 요일·휴일 구분에 따라 편차를 보이는지 미검증 상태

현행 반영 수준 (step3_features.py — 이미 적용 중인 변수)
  · 달력   : dow(순환), is_weekend, is_holiday, daytype
  · 휴일   : holiday_run_len(연휴 연속일수), days_since_holiday, days_to_holiday,
             holiday_position, is_bridge_day(샌드위치),
             is_major_holiday(명절 당일), is_major_holiday_period(명절 연휴),
             is_major_holiday_eve(명절 연휴 D−1)
  · 시차   : smp_lag_same_daytype, smp_lag_same_position
  → **학습 측면에서는 휴일 특성이 이미 투입되어 있음**

가설 — 미반영 지점은 **운영단 편의 보정**
  · 현행 편의 보정은 시간대(24)별 평균 잔차만 사용하며 **요일·휴일 구분 없음**
  · 창 K=21일에는 평일·주말이 혼재하므로,
    주말 예측에 평일 잔차가, 평일 예측에 주말 잔차가 섞여 보정 왜곡 가능

비교 구성
  | No | 구성 | 잔차 산정 단위 |
  |----|------|----------------|
  | ① | 무보정 | — |
  | ② | 현행 | 시간대(24) |
  | ③ | 2분류 | (평일·휴일) × 시간대 = 48 |
  | ④ | 3분류 | (평일·토·일공휴일) × 시간대 = 72 |
  | ⑤ | 2분류 혼합 | 동일 구분 잔차 0.5 + 전체 잔차 0.5 — 표본 부족 완화 |

방법
  · 운영 예측 이력(outputs/predict/predictions_{타깃}.csv) 사용
  · 이력의 '예측'은 보정 후 값 → 산출 시점의 감쇠계수로 **보정 전 값 역산**
    (2026-09-05 이전 γ=0.7 / 이후 γ=1.2 — 감쇠계수 개정일 반영)
  · 보정계수는 **대상일 이전 실적만** 사용 — 누출 차단

한계
  · 보정 전 역산은 근사치이며, 운영 이력 표본에 한정
  · 요일 구분 시 창 K일 내 동일 구분 일수가 감소 → 최소 이력 요건 완화 필요

실행
  python scripts\\step8_daytype_test.py                       # 진단 + 구성 비교
  python scripts\\step8_daytype_test.py --diagnose-only       # 오차 분해 진단만
  python scripts\\step8_daytype_test.py --k 21 --gamma 1.2

버전 : v1.1 (2026-09-06) — 상한(--cap) 인자화
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

CAP = 25.0
GAMMA_CHANGE_DATE = "2026-09-05"      # 감쇠계수 개정일(0.7 → 1.2)
GAMMA_OLD, GAMMA_NEW = 0.7, 1.2
DOW_KO = ["월", "화", "수", "목", "금", "토", "일"]
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {str(값):<22} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def load_calendar():
    """일자별 요일·휴일 구분 — 학습 데이터셋에서 추출"""
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c in ("date", "dow", "daytype", "is_holiday"))
    d = d.drop_duplicates("date")
    d["date"] = d["date"].astype(str)
    d["dow"] = d["dow"].astype(int)
    d["is_holiday"] = d["is_holiday"].fillna(0).astype(int)
    #  2분류 : 평일 / 휴일
    d["구분2"] = np.where(d["daytype"].astype(str) == "휴일", "휴일", "평일")
    #  3분류 : 평일 / 토 / 일·공휴일
    d["구분3"] = np.select(
        [(d["dow"] == 6) | (d["is_holiday"] == 1), d["dow"] == 5],
        ["일공휴", "토"], default="평일")
    return d[["date", "dow", "구분2", "구분3"]]


def load_hist(t, cal):
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 운영 이력 없음 → {p}")
    h = pd.read_csv(p, encoding="utf-8-sig").dropna(subset=["예측", "실적"])
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    h["date"] = h["date"].astype(str)
    h = h.merge(cal, on="date", how="left")
    return h.sort_values(["date", "hour_n"]).reset_index(drop=True)


def gamma_of(day):
    return GAMMA_NEW if day >= GAMMA_CHANGE_DATE else GAMMA_OLD


def bias_map(hist, day, K, key=None, kval=None, min_days=7):
    """대상일 이전 K일의 시간대별 평균 잔차(실적−예측). key 지정 시 동일 구분만 사용"""
    prior = hist[hist["date"] < day]
    if key is not None:
        prior = prior[prior[key] == kval]
    days = sorted(prior["date"].unique())
    if len(days) < min_days:
        return None
    sub = prior[prior["date"].isin(days[-K:])]
    return (sub["실적"] - sub["보정전"]).groupby(sub["hour_n"]).mean()


def main():
    ap = argparse.ArgumentParser(description="요일·휴일 특성 반영 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--k", type=int, default=21, help="잔차 산정 창(일)")
    ap.add_argument("--gamma", type=float, default=1.2, help="감쇠계수")
    ap.add_argument("--min-days", type=int, default=4,
                    help="구분별 최소 이력 일수(구분 적용 시 완화)")
    ap.add_argument("--cap", type=float, default=25.0,
                    help="보정폭 상한(원/kWh) — 휴일 잔차가 커 포화 여부 확인용")
    ap.add_argument("--diagnose-only", action="store_true")
    args = ap.parse_args()
    t, K, G = args.target, args.k, args.gamma
    global CAP
    CAP = args.cap

    print("=" * 78)
    print(" step8_daytype_test — 요일·휴일 특성 반영 실증 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    cal = load_calendar()
    h = load_hist(t, cal)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")

    # ── 진단 : 요일별·구분별 오차 분해 ──────────────────────
    h["오차"] = h["예측"] - h["실적"]
    h["_den"] = np.maximum(h["실적"], 20.0)

    print("\n■ ② 오차 분해 — 요일별(보정 후 운영 실적 기준)")
    print(f"  {'요일':<6}{'일수':>6}{'MAE':>9}{'편의%':>9}{'평균실적':>10}")
    print("  " + "-" * 42)
    for d_ in range(7):
        s = h[h["dow"] == d_]
        if s.empty:
            continue
        print(f"  {DOW_KO[d_]:<6}{s['date'].nunique():>6}"
              f"{s['오차'].abs().mean():>9.2f}"
              f"{(s['오차'] / s['_den']).mean() * 100:>9.2f}"
              f"{s['실적'].mean():>10.2f}")
    print("  " + "-" * 42)

    print("\n■ ③ 오차 분해 — 평일·휴일 구분")
    print(f"  {'구분':<8}{'일수':>6}{'MAE':>9}{'편의%':>9}{'평균실적':>10}")
    print("  " + "-" * 44)
    for g, s in h.groupby("구분3"):
        print(f"  {g:<8}{s['date'].nunique():>6}{s['오차'].abs().mean():>9.2f}"
              f"{(s['오차'] / s['_den']).mean() * 100:>9.2f}"
              f"{s['실적'].mean():>10.2f}")
        rep(f"오차분해({g})", f"MAE {s['오차'].abs().mean():.2f}원",
            f"편의 {(s['오차'] / s['_den']).mean() * 100:+.2f}% / "
            f"{s['date'].nunique()}일")
    print("  " + "-" * 44)

    if args.diagnose_only:
        print("\n[완료] 진단 전용 실행 — 구성 비교 생략")
        return

    # ── 보정 전 값 역산 ────────────────────────────────────
    rec = []
    h["보정전"] = h["예측"]                       # 초기값 — 순차 갱신
    for d_ in days:
        seg = h[h["date"] == d_].copy()
        base = pd.concat(rec, ignore_index=True) if rec else None
        if base is None or base["date"].nunique() < 7:
            seg["보정전"] = seg["예측"]
            seg["_적용"] = False
        else:
            prior = base[base["date"] < d_]
            sel = sorted(prior["date"].unique())[-K:]
            sub = prior[prior["date"].isin(sel)]
            b = (sub["실적"] - sub["보정전"]).groupby(sub["hour_n"]).mean()
            adj = np.clip(gamma_of(d_) * seg["hour_n"].map(b).fillna(0).values,
                          -CAP, CAP)
            seg["보정전"] = seg["예측"].values - adj
            seg["_적용"] = True
        rec.append(seg)
    hh = pd.concat(rec, ignore_index=True)
    ev = hh[hh["_적용"]].copy()
    if ev.empty:
        sys.exit("[중단] 평가 표본 없음 — 이력 부족")
    rep("평가 표본", f"{len(ev):,}행", f"{ev['date'].nunique()}일")
    rep("보정 파라미터", f"창 {K}일 / 감쇠 {G}", f"상한 ±{CAP:.0f}원")

    # ── 구성별 보정 ────────────────────────────────────────
    def apply(name, fn):
        out = []
        for d_ in sorted(ev["date"].unique()):
            seg = ev[ev["date"] == d_]
            out.append(pd.DataFrame({
                "date": seg["date"].values, "실적": seg["실적"].values,
                "구분3": seg["구분3"].values,
                "보정후": seg["보정전"].values + fn(d_, seg)}))
        r = pd.concat(out, ignore_index=True)
        r["구성"] = name
        return r

    def none_fn(d_, seg):
        return np.zeros(len(seg))

    def plain_fn(d_, seg):
        b = bias_map(hh, d_, K)
        if b is None:
            return np.zeros(len(seg))
        return np.clip(G * seg["hour_n"].map(b).fillna(0).values, -CAP, CAP)

    def split_fn(key):
        def f(d_, seg):
            kval = seg[key].iloc[0]
            b = bias_map(hh, d_, K, key=key, kval=kval, min_days=args.min_days)
            if b is None:                       # 구분별 이력 부족 시 전체 잔차 사용
                b = bias_map(hh, d_, K)
                if b is None:
                    return np.zeros(len(seg))
            return np.clip(G * seg["hour_n"].map(b).fillna(0).values, -CAP, CAP)
        return f

    def mix_fn(d_, seg):
        kval = seg["구분2"].iloc[0]
        b1 = bias_map(hh, d_, K)
        b2 = bias_map(hh, d_, K, key="구분2", kval=kval, min_days=args.min_days)
        if b1 is None:
            return np.zeros(len(seg))
        a1 = seg["hour_n"].map(b1).fillna(0).values
        a2 = seg["hour_n"].map(b2).fillna(0).values if b2 is not None else a1
        return np.clip(G * (0.5 * a1 + 0.5 * a2), -CAP, CAP)

    schemes = [("① 무보정", none_fn),
               ("② 현행(시간대)", plain_fn),
               ("③ 2분류(평일·휴일)", split_fn("구분2")),
               ("④ 3분류(평일·토·일공휴)", split_fn("구분3")),
               ("⑤ 2분류 혼합", mix_fn)]

    print("\n■ ④ 구성별 성능")
    print(f"  {'구성':<24}{'전체MAE':>9}{'평일':>8}{'토':>8}{'일공휴':>8}{'전체편의%':>10}")
    print("  " + "-" * 68)
    res = []
    for nm, fn in schemes:
        r = apply(nm, fn)
        e = r["보정후"] - r["실적"]
        den = np.maximum(r["실적"], 20.0)
        row = {"구성": nm, "전체MAE": round(float(e.abs().mean()), 2)}
        for g in ("평일", "토", "일공휴"):
            m = r["구분3"] == g
            row[g] = round(float(e[m].abs().mean()), 2) if m.any() else np.nan
        row["전체편의"] = round(float((e / den).mean() * 100), 2)
        res.append(row)
        print(f"  {nm:<24}{row['전체MAE']:>9.2f}{row['평일']:>8.2f}"
              f"{row['토']:>8.2f}{row['일공휴']:>8.2f}{row['전체편의']:>10.2f}")
        rep(nm, f"MAE {row['전체MAE']}",
            f"평일 {row['평일']} / 토 {row['토']} / 일공휴 {row['일공휴']}")
    print("  " + "-" * 68)

    df = pd.DataFrame(res).set_index("구성")
    cur = df.loc["② 현행(시간대)"]
    print("\n■ ⑤ 판정 — 현행 대비")
    for nm in df.index:
        if nm.startswith("②"):
            continue
        print(f"    {nm:<24} 전체 {df.loc[nm, '전체MAE'] - cur['전체MAE']:+.2f}원 / "
              f"휴일(일공휴) {df.loc[nm, '일공휴'] - cur['일공휴']:+.2f}원")
    best = df["전체MAE"].idxmin()
    print(f"\n    최우수(전체 MAE) : {best}")
    rep("최우수 구성", best, "전체 MAE 기준")
    print("    ※ 채택 기준 : 휴일 MAE 개선 + 평일 MAE 비악화")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_daytype_test_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        df.reset_index().to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
