# -*- coding: utf-8 -*-
"""
step8_gamma_watch.py — γ1.5 진동·2분류 주말 실효 점검 (v1.0)
==============================================================================
배경
  · 실행계획서 v1.7 / 운영매뉴얼 v2.1 : 2026. 9. 12. γ1.5 진동 점검
  · 점검 항목 : 편의 부호 전환 빈도 · 상한 포화 시간대 · 평일·휴일 2분류 주말 실효

방법
  · 운영 이력(예측 vs 실적) 일별 편의·MAE — 부호 전환은 실측 기준
  · 보정량 재현은 step6_bias_adjust 운영 파라미터 그대로 사용
    (γ 단계 · 2분류 · 이중창 혼합 일자 반영)
  · 누출 차단 : 대상일 이전 실적만 사용

실행
  python scripts\\step8_gamma_watch.py
  python scripts\\step8_gamma_watch.py --target smp_jeju

버전 : v1.0 (2026-09-12)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step6_bias_adjust import (  # noqa: E402
    GAMMA, CAP, WINDOW, WINDOW_SHORT, BLEND_W, daytype_of,
    hourly_bias, blended_bias,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")

GAMMA_STEPS = [("2026-09-07", 1.5), ("2026-09-05", 1.2), ("0000-00-00", 0.7)]
DUAL_START = "2026-09-10"
SPLIT_START = "2026-09-06"
WATCH_START = "2026-09-07"
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<28} {str(값):<28} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def gamma_of(day):
    for d0, g in GAMMA_STEPS:
        if str(day) >= d0:
            return g
    return 0.7


def mape_bias(pred, act):
    p, a = np.asarray(pred, float), np.asarray(act, float)
    m = ~(np.isnan(p) | np.isnan(a))
    if not m.any():
        return np.nan, np.nan, np.nan
    den = np.maximum(a[m], 20.0)
    err = p[m] - a[m]
    mae = float(np.abs(err).mean())
    mape = float((np.abs(err) / den).mean() * 100)
    bias = float((err / den).mean() * 100)
    return mae, mape, bias


def load_hist(t):
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 운영 이력 없음 -> {p}")
    h = pd.read_csv(p, encoding="utf-8-sig").dropna(subset=["예측", "실적"])
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    h["date"] = h["date"].astype(str).str[:10]
    h["구분2"] = h["date"].map(daytype_of)
    return h.sort_values(["date", "hour_n"]).reset_index(drop=True)


def replay_adj(h, t, day):
    """운영 파라미터로 해당일 보정량 재현 — 이력 파일의 예측은 보정 후 값"""
    g = gamma_of(day)
    if str(day) >= DUAL_START:
        b, why = blended_bias(t, day)
        mode = f"이중창 K{WINDOW}·K{WINDOW_SHORT}"
    else:
        b, why = hourly_bias(t, day)
        mode = f"단일창 K{WINDOW}"
    if b is None:
        return None, why, mode, g
    hours = h.loc[h["date"] == day, "hour_n"]
    raw = pd.Series(hours).map(b).fillna(0.0).to_numpy(float) * g
    clipped = np.clip(raw, -CAP, CAP)
    return (raw, clipped), why, mode, g


def main():
    ap = argparse.ArgumentParser(description="γ1.5 진동·2분류 주말 점검")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    args = ap.parse_args()
    t = args.target

    print("=" * 78)
    print(" step8_gamma_watch — γ1.5 진동·2분류 주말 점검 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    h = load_hist(t)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")
    watch = h[h["date"] >= WATCH_START]
    rep("점검 구간(γ1.5)", f"{WATCH_START} ~ {watch['date'].max()}",
        f"{watch['date'].nunique()}일")

    print("\n■ ② 일별 편의·부호 (운영 실측)")
    print(f"  {'일자':<12}{'요일':<6}{'구분':<6}{'MAE':>8}{'MAPE':>8}{'편의':>8}  부호")
    print("  " + "-" * 56)
    daily = []
    for d, g in h.groupby("date"):
        mae, mp, bi = mape_bias(g["예측"], g["실적"])
        wd = pd.to_datetime(d).strftime("%a")
        wd_ko = "월화수목금토일"[pd.to_datetime(d).weekday()]
        sign = "+" if bi > 0.3 else ("-" if bi < -0.3 else "0")
        daily.append({"date": d, "dow": wd_ko, "구분2": g["구분2"].iloc[0],
                      "MAE": mae, "MAPE": mp, "편의": bi, "부호": sign})
        if d >= WATCH_START:
            print(f"  {d:<12}{wd_ko:<6}{g['구분2'].iloc[0]:<6}"
                  f"{mae:>8.2f}{mp:>8.2f}{bi:>+8.2f}  {sign}")
    dd = pd.DataFrame(daily)

    w = dd[dd["date"] >= WATCH_START].copy()
    seq_all = w["부호"]
    flips = int(seq_all.ne(seq_all.shift()).iloc[1:].sum()) if len(seq_all) > 1 else 0
    seq = w.loc[w["부호"] != "0", "부호"]
    real_flips = int(seq.ne(seq.shift()).iloc[1:].sum()) if len(seq) > 1 else 0
    print("  " + "-" * 56)
    rep("γ1.5 구간 일수", f"{len(w)}일",
        f"과대(+) {(w['부호']=='+').sum()} / 과소(-) {(w['부호']=='-').sum()} / 중립 {(w['부호']=='0').sum()}")
    rep("부호 전환 횟수", f"{real_flips}회 (중립 제외)",
        f"일자 연속 전환 {flips}회")
    rep("γ1.5 구간 MAE", f"{w['MAE'].mean():.2f}원",
        f"편의 평균 {w['편의'].mean():+.2f}%")
    # 연속 동일 부호 최장
    run, mx, cur = 1, 1, None
    for s in w["부호"]:
        if s == cur and s != "0":
            run += 1
            mx = max(mx, run)
        else:
            run, cur = 1, s
    rep("최장 동일 부호 연속", f"{mx}일", "3일 이상이면 진동보다 국면 편의")

    print("\n■ ③ 보정 상한 포화 (운영 파라미터 재현)")
    sat_rows = []
    for d in sorted(watch["date"].unique()):
        replay = replay_adj(h, t, d)
        adj, why, mode, g = replay
        if adj is None:
            continue
        raw, clipped = adj
        sat = np.abs(raw) > CAP
        sat_rows.append({
            "date": d, "gamma": g, "mode": mode, "why": why,
            "n": len(raw), "sat": int(sat.sum()),
            "sat_up": int((raw > CAP).sum()),
            "sat_dn": int((raw < -CAP).sum()),
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "clip_mean": float(clipped.mean()),
        })
    sr = pd.DataFrame(sat_rows)
    if sr.empty:
        sys.exit("[중단] 보정 재현 표본 없음")
    tot, nsat = int(sr["n"].sum()), int(sr["sat"].sum())
    rate = nsat / tot if tot else 0.0
    rep("포화 행수", f"{nsat}행 / {tot}행", f"포화율 **{rate*100:.1f}%**")
    rep("포화 방향", f"상향 {int(sr['sat_up'].sum())} / 하향 {int(sr['sat_dn'].sum())}", "")
    rep("원 보정량 범위", f"{sr['raw_min'].min():+.1f} ~ {sr['raw_max'].max():+.1f}원",
        f"절대 최대 {max(abs(sr['raw_min'].min()), abs(sr['raw_max'].max())):.1f}원")
    if nsat:
        # 시간대별은 일자 루프에서 재집계
        hour_sat = []
        for d in sr["date"]:
            adj, *_ = replay_adj(h, t, d)
            if adj is None:
                continue
            raw, _c = adj
            hours = h.loc[h["date"] == d, "hour_n"].to_numpy()
            for hr, rv in zip(hours, raw):
                if abs(rv) > CAP:
                    hour_sat.append(int(hr))
        vc = pd.Series(hour_sat).value_counts().head(5)
        rep("포화 집중 시간대", ", ".join(f"{i}시({v})" for i, v in vc.items()), "상위 5")
        topd = sr.sort_values("sat", ascending=False).head(5)
        rep("포화 집중 일자", ", ".join(f"{r.date}({int(r.sat)})" for r in topd.itertuples()),
            "상위 5")
    print(f"    · 포화율 {rate*100:.1f}% — "
          f"{'상한이 실질 제약(10% 이상)' if rate >= 0.10 else '제약 경미(10% 미만)'}")

    print("\n■ ④ 평일·휴일 2분류 주말 실효 (적용일 2026-09-06~)")
    post = dd[dd["date"] >= SPLIT_START].copy()
    pre = dd[(dd["date"] < SPLIT_START) & (dd["date"] >= "2026-08-20")].copy()

    def blk(name, frame):
        if frame.empty:
            return
        print(f"  [{name}]")
        for k, sub in (("평일", frame[frame["구분2"] == "평일"]),
                       ("휴일", frame[frame["구분2"] == "휴일"])):
            if sub.empty:
                print(f"    {k}: 표본 없음")
                continue
            print(f"    {k} {len(sub)}일  MAE {sub['MAE'].mean():.2f}  "
                  f"편의 {sub['편의'].mean():+.2f}%  "
                  f"일자 {', '.join(sub['date'].str[5:] + sub['dow'])}")
            rep(f"{name} {k}",
                f"{len(sub)}일 / MAE {sub['MAE'].mean():.2f} / 편의 {sub['편의'].mean():+.2f}%",
                ", ".join(sub["date"]))

    blk("적용 전(8.20~9.5)", pre)
    blk("적용 후(9.6~)", post)
    hol = post[post["구분2"] == "휴일"]
    wd = post[post["구분2"] == "평일"]
    if len(hol) >= 2 and len(wd) >= 2:
        gap_pre = (pre[pre["구분2"] == "휴일"]["MAE"].mean()
                   - pre[pre["구분2"] == "평일"]["MAE"].mean()) if not pre.empty else np.nan
        gap_post = hol["MAE"].mean() - wd["MAE"].mean()
        rep("휴일-평일 MAE 격차",
            f"적용전 {gap_pre:.2f} → 적용후 {gap_post:.2f}원",
            "격차 축소가 실효")
        # 토 표본
        sat = hol[hol["dow"] == "토"]
        sun = hol[hol["dow"] == "일"]
        rep("토 표본", f"{len(sat)}일 MAE {sat['MAE'].mean():.2f}" if len(sat) else "0일",
            ", ".join(sat["date"]) if len(sat) else "3분류 재실증(9.27) 이관")
        rep("일 표본", f"{len(sun)}일 MAE {sun['MAE'].mean():.2f}" if len(sun) else "0일",
            ", ".join(sun["date"]))

    print("\n■ ⑤ 판정")
    vib = "국면 편의(연속 동부호)" if mx >= 3 and real_flips <= 2 else (
        "진동(빈번 전환)" if real_flips >= 3 else "관찰 유지")
    cap_j = "상한 상향 검토" if rate >= 0.10 else "상한 ±25원 유지"
    if len(hol) < 2:
        split_j = "표본 부족 — 9.27 재실증으로 이관"
    elif (hol["MAE"].mean() < pre[pre["구분2"] == "휴일"]["MAE"].mean()
          if not pre[pre["구분2"] == "휴일"].empty else True):
        split_j = "2분류 유지(휴일 MAE 개선 또는 악화 없음)"
    else:
        split_j = "휴일 악화 — 원인 기록, 2분류 유지 후 9.27 재판정"
    print(f"    · 진동 : {vib}")
    print(f"    · 상한 : {cap_j} (포화율 {rate*100:.1f}%)")
    print(f"    · 2분류 : {split_j}")
    print(f"    · γ 1.5 : {'재조정 검토' if (w['MAE'].mean() > 14 or abs(w['편의'].mean()) > 8) else '유지'}"
          f" (구간 MAE {w['MAE'].mean():.2f}원 / 편의 {w['편의'].mean():+.2f}%)")
    rep("판정_진동", vib, "")
    rep("판정_상한", cap_j, f"포화율 {rate*100:.1f}%")
    rep("판정_2분류", split_j, "")
    rep("판정_감쇠", "재조정 검토" if (w["MAE"].mean() > 14 or abs(w["편의"].mean()) > 8) else "γ 1.5 유지",
        f"MAE {w['MAE'].mean():.2f} / 편의 {w['편의'].mean():+.2f}%")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_gamma_watch_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        dd.to_csv(f, index=False)
        f.write("\n")
        sr.to_csv(f, index=False)
    print(f"\n[완료] 로그 -> {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
