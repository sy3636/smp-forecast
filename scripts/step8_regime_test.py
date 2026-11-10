# -*- coding: utf-8 -*-
"""
step8_regime_test.py — 수준 전환 국면 대응 실증 (v1.0)
==============================================================================
배경
  · 2026년 운영 실적에서 **수준 전환 국면의 양방향 실패** 확인
    - 8월 급등 전환(실적평균 142.5 → 162.6원) : 과소예측, 9.1 편의 −6.20%
    - 9월 급락 전환(160원대 → 108원대)       : 과대예측, 9월 편의 +9.69%
  · 원인 : 주 1회 재학습이 전환에 후행하고, 편의 보정 창(21일)이 전환기에 둔감
  · 선행 실증 : 재학습 주기 단축(7일 이하 포화·반전, 기각),
               수준 편차 변수(운영 조건에서 순효과 소멸, 기각)
    → **학습 측면의 수단은 한계 확인**. 운영단 보정의 적응성 개선이 남은 경로

가설
  · 수준 전환이 감지된 시점에는 **짧은 창·강한 감쇠**로 잔차를 신속 반영하고,
    평시에는 현행(창 21일·γ 1.5)을 유지하는 **적응형 보정**이 우월함
  · v1.4(2026.9.10) : 현행 γ 1.5 반영, 역산에 산출 시점 γ 적용, 감쇠 후보 1.0~2.5로 확장

전환 감지
  · 지표 : 직전 3일 일평균 실적 ÷ 직전 4~14일 일평균 실적 − 1
  · 임계 : |지표| > θ (기본 0.10) → 전환 국면으로 판정
  · 대상일 이전 실적만 사용 — 누출 없음

비교 구성
  | No | 구성 | 창 K | 감쇠 γ |
  |----|------|------|--------|
  | ①  | 무보정 | — | — |
  | ②  | 현행 | 21 | **1.5** |
  | ③  | 짧은 창 | 7 | 1.5 |
  | ④  | **적응형** | 전환 5 / 평시 21 | 전환 **2.0** / 평시 1.5 |
  | ⑤  | 이중창 혼합 | 21·5 평균 | 1.5 |

방법
  · 운영 예측 이력(outputs/predict/predictions_{타깃}.csv)을 사용
  · 이력의 '예측'은 보정 후 값이므로, 현행 파라미터를 재현하여 **보정 전 값을 역산**
  · 각 구성의 보정을 보정 전 값에 적용하여 MAE·편의를 비교
  · 보정계수는 **대상일 이전 실적만** 사용 — 누출 차단

한계
  · 보정 전 역산은 현행 파라미터 재현에 근거한 **근사치**
  · 운영 이력 표본에 한정되며, 학습 모형 변경 효과는 포함하지 않음

실행
  python scripts\\step8_regime_test.py
  python scripts\\step8_regime_test.py --target smp_jeju --theta 0.08
  python scripts\\step8_regime_test.py --sweep        # K·γ 격자 탐색
  python scripts\\step8_regime_test.py --sweep --k-list 7,10,14,21 --g-list 0.8,1.0,1.2,1.5
  python scripts\\step8_regime_test.py --validate     # 전반 탐색 → 후반 검증(과적합 확인)

버전 : v1.4 (2026-09-10) — 현행 γ1.5 반영·시점별 역산·감쇠 후보 확장
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

#  현행 운영 파라미터 — 2026.9.10 갱신(종전 도구는 γ0.7 고정으로 판정 왜곡)
#   · γ 이력 : 2026.9.5 0.7→1.2 / 2026.9.7 1.2→1.5
#   · 보정 전 역산은 **산출 시점 γ**를 적용해야 정확 — GAMMA_STEPS 사용
CUR_K, CUR_G, CAP, MIN_DAYS = 21, 1.5, 25.0, 7
GAMMA_STEPS = [("2026-09-07", 1.5), ("2026-09-05", 1.2), ("0000-00-00", 0.7)]


def gamma_of(day):
    """산출 시점 감쇠계수 — 역산 정확도 확보"""
    for d0, g in GAMMA_STEPS:
        if str(day) >= d0:
            return g
    return 0.7
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<24} {str(값):<24} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def load_hist(t):
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 운영 이력 없음 → {p}")
    h = pd.read_csv(p, encoding="utf-8-sig")
    need = {"date", "hour_n", "예측", "실적"}
    if not need.issubset(h.columns):
        sys.exit("[중단] 이력 형식 불일치")
    h = h.dropna(subset=["예측", "실적"])
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    h["date"] = h["date"].astype(str)
    return h.sort_values(["date", "hour_n"]).reset_index(drop=True)


def bias_map(hist, day, K, min_days=MIN_DAYS):
    """대상일 이전 K일의 시간대별 평균 잔차(실적−예측)"""
    prior = hist[hist["date"] < day]
    days = sorted(prior["date"].unique())
    if len(days) < min_days:
        return None
    sel = days[-K:]
    sub = prior[prior["date"].isin(sel)]
    return (sub["실적"] - sub["예측"]).groupby(sub["hour_n"]).mean()


def load_daily(t):
    """일평균 실적 시계열 — 전환 감지용

    비고 : 예측 이력은 산출 누락으로 일자가 불연속하므로,
           전환 감지는 **연속 실적 시계열**(smp_dataset_hourly)을 기준으로 산정함
    """
    if not os.path.exists(DATASET):
        return None
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c in ("date", t))
    if t not in d.columns:
        return None
    d["date"] = d["date"].astype(str)
    return d.dropna(subset=[t]).groupby("date")[t].mean().sort_index()


def regime_flag(daily, day, theta):
    """수준 전환 감지 — 직전 3일 대비 직전 4~14일 일평균 비율(연속 실적 기준)"""
    if daily is None:
        return False, np.nan
    prior = daily[daily.index < day]
    if len(prior) < 14:
        return False, np.nan
    recent = prior.iloc[-3:].mean()
    base = prior.iloc[-14:-3].mean()
    if not np.isfinite(base) or base == 0:
        return False, np.nan
    r = recent / base - 1.0
    return (abs(r) > theta), float(r)


def main():
    ap = argparse.ArgumentParser(description="수준 전환 국면 대응 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--theta", type=float, default=0.10,
                    help="전환 감지 임계(기본 0.10 = 10%%)")
    ap.add_argument("--sweep", action="store_true",
                    help="K·γ 격자 탐색 — 최적 창 길이의 국소성 여부 확인")
    ap.add_argument("--k-list", default="3,5,7,10,14,21,28",
                    help="격자 창 길이 목록(쉼표 구분)")
    ap.add_argument("--g-list", default="1.0,1.2,1.5,1.8,2.0,2.5",
                    help="격자 감쇠 목록(쉼표 구분)")
    ap.add_argument("--validate", action="store_true",
                    help="시간 분할 검증 — 전반 격자 탐색 → 후반 적용(과적합 확인)")
    args = ap.parse_args()
    t, theta = args.target, args.theta

    print("=" * 76)
    print(" step8_regime_test — 수준 전환 국면 대응 실증 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 76)

    h = load_hist(t)
    daily = load_daily(t)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")
    rep("전환 감지 임계", f"|3일/직전11일 − 1| > {theta:.2f}",
        "연속 실적 시계열 기준")

    #  보정 전 값 역산 — 현행 파라미터(K=21, γ=0.7) 재현
    rec = []
    for d in days:
        b = bias_map(h, d, CUR_K)
        seg = h[h["date"] == d].copy()
        if b is None:
            seg["보정전"] = seg["예측"]
            seg["_적용"] = False
        else:
            adj = np.clip(gamma_of(d) * seg["hour_n"].map(b).fillna(0).values,
                          -CAP, CAP)
            seg["보정전"] = seg["예측"].values - adj
            seg["_적용"] = True
        flag, r = regime_flag(daily, d, theta)
        seg["_전환"] = flag
        seg["_비율"] = r
        rec.append(seg)
    hh = pd.concat(rec, ignore_index=True)
    ev = hh[hh["_적용"]].copy()
    if ev.empty:
        sys.exit("[중단] 평가 가능 표본 없음 — 이력 부족")

    n_rg = ev[ev["_전환"]]["date"].nunique()
    rep("평가 표본", f"{len(ev):,}행", f"{ev['date'].nunique()}일")
    rep("전환 국면 판정", f"{n_rg}일", f"{n_rg / max(ev['date'].nunique(), 1) * 100:.0f}%")
    if n_rg:
        rr = ev[ev["_전환"]].groupby("date")["_비율"].first()
        print("\n  전환 판정일 :")
        for k, v in rr.items():
            print(f"    {k}  수준변화 {v:+.1%}")

    #  구성별 보정 적용
    def apply_scheme(name, fn):
        out = []
        for d in sorted(ev["date"].unique()):
            seg = ev[ev["date"] == d]
            adj = fn(d, seg)
            out.append(pd.DataFrame({
                "date": seg["date"].values, "실적": seg["실적"].values,
                "보정후": seg["보정전"].values + adj,
                "전환": seg["_전환"].values}))
        r = pd.concat(out, ignore_index=True)
        r["구성"] = name
        return r

    def none_fn(d, seg):
        return np.zeros(len(seg))

    def fixed_fn(K, G):
        def f(d, seg):
            b = bias_map(h, d, K)
            if b is None:
                return np.zeros(len(seg))
            return np.clip(G * seg["hour_n"].map(b).fillna(0).values, -CAP, CAP)
        return f

    def adaptive_fn(d, seg):
        flag = bool(seg["_전환"].iloc[0])
        #  2026.9.10 갱신 : 전환 시 γ는 현행(1.5) **초과** 값이어야 의미 있음
        K, G = (5, 2.0) if flag else (CUR_K, CUR_G)
        b = bias_map(h, d, K, min_days=3 if flag else MIN_DAYS)
        if b is None:
            return np.zeros(len(seg))
        return np.clip(G * seg["hour_n"].map(b).fillna(0).values, -CAP, CAP)

    def blend_fn(d, seg):
        b1, b2 = bias_map(h, d, CUR_K), bias_map(h, d, 5, min_days=3)
        if b1 is None:
            return np.zeros(len(seg))
        a1 = seg["hour_n"].map(b1).fillna(0).values
        a2 = seg["hour_n"].map(b2).fillna(0).values if b2 is not None else a1
        return np.clip(CUR_G * (0.5 * a1 + 0.5 * a2), -CAP, CAP)

    schemes = [
        ("① 무보정", none_fn),
        (f"② 현행 K{CUR_K}·γ{CUR_G}", fixed_fn(CUR_K, CUR_G)),
        (f"③ 짧은창 K7·γ{CUR_G}", fixed_fn(7, CUR_G)),
        ("④ 적응형", adaptive_fn),
        ("⑤ 이중창 혼합", blend_fn),
    ]

    print("\n■ ② 구성별 성능")
    print(f"  {'구성':<18}{'전체MAE':>9}{'전환MAE':>9}{'평시MAE':>9}"
          f"{'전체편의':>10}{'전환편의':>10}")
    print("  " + "-" * 66)
    res = []
    for nm, fn in schemes:
        r = apply_scheme(nm, fn)
        e = r["보정후"] - r["실적"]
        den = np.maximum(r["실적"], 20.0)
        rg, nm_ = r["전환"], ~r["전환"]
        row = {
            "구성": nm,
            "전체MAE": round(float(e.abs().mean()), 2),
            "전환MAE": round(float(e[rg].abs().mean()), 2) if rg.any() else np.nan,
            "평시MAE": round(float(e[nm_].abs().mean()), 2) if nm_.any() else np.nan,
            "전체편의": round(float((e / den).mean() * 100), 2),
            "전환편의": round(float((e[rg] / den[rg]).mean() * 100), 2) if rg.any() else np.nan,
        }
        res.append(row)
        print(f"  {nm:<18}{row['전체MAE']:>9.2f}{row['전환MAE']:>9.2f}"
              f"{row['평시MAE']:>9.2f}{row['전체편의']:>9.2f}%{row['전환편의']:>9.2f}%")
        rep(nm, f"MAE {row['전체MAE']} (전환 {row['전환MAE']})",
            f"편의 {row['전체편의']}% / 전환 {row['전환편의']}%")
    print("  " + "-" * 66)

    if args.sweep:
        print("\n■ ②-2 K·γ 격자 탐색")
        print(f"  {'K':>4}{'γ':>7}{'전체MAE':>10}{'전환MAE':>10}{'평시MAE':>10}")
        print("  " + "-" * 41)
        sw = []
        k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
        g_list = [float(x) for x in args.g_list.split(",") if x.strip()]
        for K in k_list:
            for G in g_list:
                r = apply_scheme(f"K{K}·g{G}", fixed_fn(K, G))
                e = r["보정후"] - r["실적"]
                rg, nm_ = r["전환"], ~r["전환"]
                row = {"K": K, "γ": G,
                       "전체MAE": round(float(e.abs().mean()), 2),
                       "전환MAE": round(float(e[rg].abs().mean()), 2) if rg.any() else np.nan,
                       "평시MAE": round(float(e[nm_].abs().mean()), 2) if nm_.any() else np.nan}
                sw.append(row)
                print(f"  {K:>4}{G:>7.1f}{row['전체MAE']:>10.2f}"
                      f"{row['전환MAE']:>10.2f}{row['평시MAE']:>10.2f}")
        print("  " + "-" * 41)
        sdf = pd.DataFrame(sw)
        b = sdf.loc[sdf["전체MAE"].idxmin()]
        print(f"    최소 전체MAE : K={int(b['K'])}·γ={b['γ']}  {b['전체MAE']:.2f}원")
        rep("격자 최적", f"K={int(b['K'])}·γ={b['γ']}", f"전체 MAE {b['전체MAE']:.2f}원")

    if args.validate:
        print("\n■ ②-3 시간 분할 검증 — 과적합 확인")
        k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
        g_list = [float(x) for x in args.g_list.split(",") if x.strip()]
        edays = sorted(ev["date"].unique())
        cut = len(edays) // 2
        d_tr, d_te = set(edays[:cut]), set(edays[cut:])
        rep("탐색 구간(전반)", f"{edays[0]} ~ {edays[cut - 1]}", f"{cut}일")
        rep("검증 구간(후반)", f"{edays[cut]} ~ {edays[-1]}", f"{len(edays) - cut}일")

        def mae_on(fn, dset):
            r = apply_scheme("_", fn)
            m = r["date"].isin(dset)
            return float((r.loc[m, "보정후"] - r.loc[m, "실적"]).abs().mean())

        best, bm = None, np.inf
        for K in k_list:
            for G in g_list:
                m = mae_on(fixed_fn(K, G), d_tr)
                if m < bm:
                    bm, best = m, (K, G)
        cur_tr = mae_on(fixed_fn(CUR_K, CUR_G), d_tr)
        cur_te = mae_on(fixed_fn(CUR_K, CUR_G), d_te)
        bst_te = mae_on(fixed_fn(*best), d_te)
        print(f"  {'구간':<12}{f'현행 K{CUR_K}·γ{CUR_G}':>16}"
              f"{'전반 최적':>16}{'차이':>10}")
        print("  " + "-" * 54)
        print(f"  {'전반(탐색)':<12}{cur_tr:>16.2f}{bm:>16.2f}{bm - cur_tr:>10.2f}")
        print(f"  {'후반(검증)':<12}{cur_te:>16.2f}{bst_te:>16.2f}{bst_te - cur_te:>10.2f}")
        print("  " + "-" * 54)
        rep("전반 최적 파라미터", f"K={best[0]}·γ={best[1]}", f"전반 MAE {bm:.2f}원")
        rep("후반 검증 결과", f"{bst_te:.2f}원",
            f"현행 {cur_te:.2f}원 대비 {bst_te - cur_te:+.2f}원")
        print(f"    → 후반 개선 시 일반화 확인 / 악화 시 과적합 판정")

    print("\n■ ③ 판정")
    df = pd.DataFrame(res).set_index("구성")
    cur = df.loc[f"② 현행 K{CUR_K}·γ{CUR_G}"]
    best = df["전체MAE"].idxmin()
    for nm in df.index:
        if nm.startswith("②"):
            continue
        d1 = df.loc[nm, "전체MAE"] - cur["전체MAE"]
        d2 = df.loc[nm, "전환MAE"] - cur["전환MAE"]
        print(f"    {nm:<18} 전체 {d1:+.2f}원 / 전환 {d2:+.2f}원")
        rep(f"현행 대비 {nm}", f"전체 {d1:+.2f}원", f"전환 {d2:+.2f}원")
    print(f"\n    최우수(전체 MAE) : {best}")
    rep("최우수 구성", best, "전체 MAE 기준")
    print("    ※ 채택 기준 : 전환 국면 MAE 개선 + 평시 MAE 비악화")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_regime_test_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        df.reset_index().to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
