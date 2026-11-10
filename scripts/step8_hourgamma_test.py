# -*- coding: utf-8 -*-
"""
step8_hourgamma_test.py — 시간대별 감쇠 차등 실증 (v1.0)
==============================================================================
배경
  · 2026. 9. 7. 진단에서 **편의의 시간대 편중** 확인
    | 구간 | MAE | 편의 |
    |------|----:|-----:|
    | 주간 10~16시 | 17.80원 | **+10.60%** |
    | 그 외 시간대 | 7.60원 | −3.09% |
  · 월별 주간 편의 : 2026-08 +1.02% → **2026-09 +16.65%**
  · 계절 요인(냉방수요 감소)·기온계 변수 투입은 **기각** — 원인은 수준 전환 미추종
  · 현행 편의 보정은 감쇠 γ=1.2를 **전 시간대 균일 적용**

가설
  · 편의가 큰 주간 시간대에 **감쇠를 강화**하면 전환 국면 대응이 개선됨

비교 구성 (창 21일 · 평일·휴일 2분류 · 상한 ±25원 — 현행 운영 조건)
  | No | 구성 | 주간(10~16시) γ | 그 외 γ |
  |----|------|-----------------|---------|
  | ①  | 무보정 | — | — |
  | ②  | 현행 균일 | 1.2 | 1.2 |
  | ③  | 주간 강화 | 1.5 | 1.2 |
  | ④  | 주간 강화 | 1.8 | 1.2 |
  | ⑤  | 주간 강화 | 2.0 | 1.2 |
  | ⑥  | 균일 상향(대조) | 1.5 | 1.5 |

  · ⑥은 **주간 특화 효과와 단순 상향 효과를 분리**하기 위한 대조군

방법
  · 운영 이력의 '예측'은 보정 후 값 → 산출 시점 감쇠계수로 보정 전 값 역산
    (2026-09-05 이전 γ=0.7 / 이후 γ=1.2)
  · 보정계수는 대상일 이전 실적만 사용 — 누출 차단
  · 구분별 이력 4일 미만 시 전체 잔차로 대체(운영 로직과 동일)

판정
  · 채택 기준 : **주간 MAE 개선 + 야간 MAE 비악화 + 전체 MAE 개선**

실행
  python scripts\\step8_hourgamma_test.py
  python scripts\\step8_hourgamma_test.py --target smp_jeju --k 21

버전 : v1.0 (2026-09-07)
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
MIN_DAYS_SPLIT = 4
GAMMA_CHANGE_DATE = "2026-09-05"
GAMMA_OLD, GAMMA_NEW = 0.7, 1.2
DAY_HOURS = list(range(10, 17))
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<24} {str(값):<22} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def bias(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(((p[m] - a[m]) / np.maximum(a[m], 20.0)).mean() * 100)


def load_calendar():
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c in ("date", "daytype"))
    d = d.drop_duplicates("date")
    d["date"] = d["date"].astype(str)
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
    h["date"] = h["date"].astype(str)
    return (h.merge(cal, on="date", how="left")
            .sort_values(["date", "hour_n"]).reset_index(drop=True))


def gamma_of(day):
    return GAMMA_NEW if day >= GAMMA_CHANGE_DATE else GAMMA_OLD


def bias_map(hist, day, K, kval=None, col="보정전"):
    """대상일 이전 K일의 시간대별 평균 잔차 — kval 지정 시 동일 구분만"""
    prior = hist[hist["date"] < day]
    if kval is not None:
        sub = prior[prior["구분2"] == kval]
        if sub["date"].nunique() >= MIN_DAYS_SPLIT:
            prior = sub
    days = sorted(prior["date"].unique())
    if len(days) < 7:
        return None
    s = prior[prior["date"].isin(days[-K:])]
    return (s["실적"] - s[col]).groupby(s["hour_n"]).mean()


def main():
    ap = argparse.ArgumentParser(description="시간대별 감쇠 차등 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--k", type=int, default=21, help="잔차 산정 창(일)")
    ap.add_argument("--day-gammas", default="1.5,1.8,2.0",
                    help="주간 감쇠 후보(쉼표 구분)")
    args = ap.parse_args()
    t, K = args.target, args.k

    print("=" * 78)
    print(" step8_hourgamma_test — 시간대별 감쇠 차등 실증 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    cal = load_calendar()
    h = load_hist(t, cal)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")
    rep("보정 조건", f"창 {K}일 / 상한 ±{CAP:.0f}원", "평일·휴일 2분류(운영 동일)")
    rep("주간 정의", "10~16시", f"{len(DAY_HOURS)}개 시간대")

    #  보정 전 역산 — 순차 갱신
    rec = []
    h["보정전"] = h["예측"]
    for d in days:
        seg = h[h["date"] == d].copy()
        base = pd.concat(rec, ignore_index=True) if rec else None
        if base is None or base["date"].nunique() < 7:
            seg["_적용"] = False
        else:
            b = bias_map(base, d, K, seg["구분2"].iloc[0])
            if b is None:
                seg["_적용"] = False
            else:
                adj = np.clip(gamma_of(d) * seg["hour_n"].map(b).fillna(0).values,
                              -CAP, CAP)
                seg["보정전"] = seg["예측"].values - adj
                seg["_적용"] = True
        rec.append(seg)
    hh = pd.concat(rec, ignore_index=True)
    ev = hh[hh["_적용"]].copy()
    if ev.empty:
        sys.exit("[중단] 평가 표본 없음")
    rep("평가 표본", f"{len(ev):,}행", f"{ev['date'].nunique()}일")

    #  구성별 적용
    def apply(g_day, g_night):
        out = []
        for d in sorted(ev["date"].unique()):
            seg = ev[ev["date"] == d]
            b = bias_map(hh, d, K, seg["구분2"].iloc[0])
            if b is None:
                adj = np.zeros(len(seg))
            else:
                raw = seg["hour_n"].map(b).fillna(0).values
                g = np.where(seg["hour_n"].isin(DAY_HOURS).values, g_day, g_night)
                adj = np.clip(g * raw, -CAP, CAP)
            out.append(pd.DataFrame({
                "hour_n": seg["hour_n"].values, "실적": seg["실적"].values,
                "보정후": seg["보정전"].values + adj}))
        return pd.concat(out, ignore_index=True)

    day_gs = [float(x) for x in args.day_gammas.split(",") if x.strip()]
    schemes = [("① 무보정", None, None),
               ("② 현행 균일 1.2", 1.2, 1.2)]
    for g in day_gs:
        schemes.append((f"③ 주간 {g} / 야간 1.2", g, 1.2))
    schemes.append((f"⑥ 균일 {day_gs[0]}(대조)", day_gs[0], day_gs[0]))

    print("\n■ ② 구성별 성능")
    print(f"  {'구성':<22}{'전체MAE':>9}{'주간MAE':>9}{'야간MAE':>9}"
          f"{'주간편의%':>10}{'전체편의%':>10}")
    print("  " + "-" * 70)
    res = {}
    for nm, gd, gn in schemes:
        if gd is None:
            r = pd.DataFrame({"hour_n": ev["hour_n"].values,
                              "실적": ev["실적"].values,
                              "보정후": ev["보정전"].values})
        else:
            r = apply(gd, gn)
        di = r["hour_n"].isin(DAY_HOURS)
        row = {"전체MAE": mae(r["보정후"], r["실적"]),
               "주간MAE": mae(r.loc[di, "보정후"], r.loc[di, "실적"]),
               "야간MAE": mae(r.loc[~di, "보정후"], r.loc[~di, "실적"]),
               "주간편의": bias(r.loc[di, "보정후"], r.loc[di, "실적"]),
               "전체편의": bias(r["보정후"], r["실적"])}
        res[nm] = row
        print(f"  {nm:<22}{row['전체MAE']:>9.2f}{row['주간MAE']:>9.2f}"
              f"{row['야간MAE']:>9.2f}{row['주간편의']:>10.2f}{row['전체편의']:>10.2f}")
        rep(nm, f"MAE {row['전체MAE']:.2f}",
            f"주간 {row['주간MAE']:.2f} / 야간 {row['야간MAE']:.2f} / "
            f"주간편의 {row['주간편의']:+.2f}%")
    print("  " + "-" * 70)

    cur = res.get("② 현행 균일 1.2")
    print("\n■ ③ 판정 — 현행 대비")
    for nm, row in res.items():
        if nm.startswith("②"):
            continue
        print(f"    {nm:<22} 전체 {row['전체MAE'] - cur['전체MAE']:+.2f}원 / "
              f"주간 {row['주간MAE'] - cur['주간MAE']:+.2f} / "
              f"야간 {row['야간MAE'] - cur['야간MAE']:+.2f} / "
              f"주간편의 {row['주간편의'] - cur['주간편의']:+.2f}%p")
        rep(f"현행 대비 {nm}", f"전체 {row['전체MAE'] - cur['전체MAE']:+.2f}원",
            f"주간 {row['주간MAE'] - cur['주간MAE']:+.2f} / "
            f"야간 {row['야간MAE'] - cur['야간MAE']:+.2f}")
    best = min(res, key=lambda k: res[k]["전체MAE"])
    print(f"\n    최우수(전체 MAE) : {best}")
    rep("최우수 구성", best, "전체 MAE 기준")
    print("    ※ 채택 기준 : 주간 MAE 개선 + 야간 MAE 비악화 + 전체 MAE 개선")
    print("    ※ ⑥ 균일 대조군 대비 우월해야 **주간 특화의 순효과** 인정")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_hourgamma_test_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        pd.DataFrame(res).T.reset_index().to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
