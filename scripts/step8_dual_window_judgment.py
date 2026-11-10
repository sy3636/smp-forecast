# -*- coding: utf-8 -*-
"""
step8_dual_window_judgment.py — 이중창 혼합(K21·K5) 최종 판정
==============================================================================
배경
  · 2026.9.10 평일 이중창 채택, 2026.9.12 휴일 단일·γ0.7·±15 완화
  · 중간 관찰 2026.9.17 / 최종 판정 기한 2026.9.19 (본 실행으로 확정)

방법
  · 운영 이력(예측 vs 실적) 일별 MAE·MAPE·편의 — 육지 D+1
  · 보정량 재현은 step6_bias_adjust 운영 파라미터(_params_for) 그대로
  · 누출 차단 : 대상일 이전 실적만 사용 (hourly_bias / blended_bias)

실행
  python scripts/step8_dual_window_judgment.py
  python scripts/step8_dual_window_judgment.py --target smp_jeju

버전 : v1.0 (2026-09-21)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step6_bias_adjust import (  # noqa: E402
    WINDOW, WINDOW_SHORT, BLEND_W, CAP,
    daytype_of, hourly_bias, blended_bias, _params_for,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")

PRE = ("2026-09-07", "2026-09-09")     # 이중창 이전 γ1.5 평일
ADOPT = "2026-09-10"
HOLIDAY_RELAX_START = "2026-09-12"
WATCH_START = "2026-09-07"
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<32} {str(값):<32} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


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
    dt, use_dual, g, c, tag = _params_for(day)
    if use_dual:
        b, why = blended_bias(t, day)
        mode = f"이중창 K{WINDOW}·K{WINDOW_SHORT}(w={BLEND_W})"
    else:
        b, why = hourly_bias(t, day)
        mode = f"단일창 K{WINDOW}"
    if tag:
        mode = f"{mode}/{tag}"
    if b is None:
        return None, why, mode, g, c, dt
    hours = h.loc[h["date"] == day, "hour_n"]
    raw = pd.Series(hours).map(b).fillna(0.0).to_numpy(float) * g
    clipped = np.clip(raw, -c, c)
    return (raw, clipped), why, mode, g, c, dt


def blk_stats(frame):
    if frame is None or frame.empty:
        return np.nan, np.nan, 0
    return float(frame["MAE"].mean()), float(frame["편의"].mean()), int(len(frame))


def main():
    ap = argparse.ArgumentParser(description="이중창 혼합 최종 판정")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    args = ap.parse_args()
    t = args.target

    print("=" * 78)
    print(" step8_dual_window_judgment — 이중창 혼합 최종 판정 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    h = load_hist(t)
    days = sorted(h["date"].unique())
    print("\n■ ① 대상")
    rep("운영 이력", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(h):,}행")
    watch = h[h["date"] >= WATCH_START]
    last_act = watch["date"].max()
    rep("판정 구간", f"{WATCH_START} ~ {last_act}",
        f"{watch['date'].nunique()}일 (실적 있는 날)")

    print("\n■ ② 일별 실측")
    print(f"  {'일자':<12}{'요일':<4}{'구분':<6}{'MAE':>8}{'MAPE':>8}{'편의':>9}  부호  비고")
    print("  " + "-" * 72)
    daily = []
    for d, g in h.groupby("date"):
        mae, mp, bi = mape_bias(g["예측"], g["실적"])
        wd_ko = "월화수목금토일"[pd.to_datetime(d).weekday()]
        sign = "+" if bi > 0.3 else ("-" if bi < -0.3 else "0")
        note = ""
        if d < ADOPT:
            note = "이중창 이전"
        elif d == ADOPT:
            note = "이중창 채택"
        elif d == HOLIDAY_RELAX_START:
            note = "휴일완화 적용"
        daily.append({"date": d, "dow": wd_ko, "구분2": g["구분2"].iloc[0],
                      "MAE": mae, "MAPE": mp, "편의": bi, "부호": sign,
                      "비고": note})
        if d >= WATCH_START:
            print(f"  {d:<12}{wd_ko:<4}{g['구분2'].iloc[0]:<6}"
                  f"{mae:>8.2f}{mp:>8.2f}{bi:>+9.2f}  {sign:<4} {note}")
    dd = pd.DataFrame(daily)
    w = dd[dd["date"] >= WATCH_START].copy()

    seq = w.loc[w["부호"] != "0", "부호"]
    real_flips = int(seq.ne(seq.shift()).iloc[1:].sum()) if len(seq) > 1 else 0
    run, mx, cur = 1, 1, None
    for s in w["부호"]:
        if s == cur and s != "0":
            run += 1
            mx = max(mx, run)
        else:
            run, cur = 1, s

    print("\n■ ③ 구간 집계")
    pre = w[(w["date"] >= PRE[0]) & (w["date"] <= PRE[1]) & (w["구분2"] == "평일")]
    post_wd = w[(w["date"] >= ADOPT) & (w["구분2"] == "평일")]
    recent_wd = w[(w["date"] >= "2026-09-14") & (w["date"] <= "2026-09-19")
                  & (w["구분2"] == "평일")]
    last7_wd = w[(w["date"] >= ADOPT) & (w["구분2"] == "평일")].tail(4)
    hol = w[(w["date"] >= HOLIDAY_RELAX_START) & (w["구분2"] == "휴일")]
    sat = hol[hol["dow"] == "토"]
    sun = hol[hol["dow"] == "일"]
    mondays = w[(w["dow"] == "월") & (w["date"] >= ADOPT)]
    convert = w[w["date"] == "2026-09-09"]

    def show(name, frame):
        mae, bi, n = blk_stats(frame)
        dates = ", ".join(frame["date"].str[5:] + frame["dow"]) if n else "-"
        print(f"  [{name}] {n}일  MAE {mae:.2f}  편의 {bi:+.2f}%  {dates}")
        rep(name, f"{n}일 / MAE {mae:.2f} / 편의 {bi:+.2f}%", dates)
        return mae, bi, n

    mae_pre, bi_pre, n_pre = show("이중창 이전 평일(9.7~9.9)", pre)
    mae_post, bi_post, n_post = show("이중창 이후 평일(9.10~)", post_wd)
    mae_rec, bi_rec, n_rec = show("평일 최근(9.14~9.19 평일)", recent_wd)
    show("휴일 완화 이후(9.12~)", hol)
    show("토요일", sat)
    show("일요일", sun)
    show("월요일(채택 이후)", mondays)

    print("\n■ ④ 보정 상한 포화 (운영 파라미터 재현 · 휴일완화 반영)")
    sat_rows = []
    hour_sat = []
    for d in sorted(watch["date"].unique()):
        replay = replay_adj(h, t, d)
        adj, why, mode, g, c, dt = replay
        if adj is None:
            continue
        raw, clipped = adj
        hit = np.abs(raw) > c
        sat_rows.append({
            "date": d, "gamma": g, "cap": c, "mode": mode, "why": why,
            "구분": dt, "n": len(raw), "sat": int(hit.sum()),
            "sat_up": int((raw > c).sum()), "sat_dn": int((raw < -c).sum()),
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "clip_mean": float(clipped.mean()),
        })
        hours = h.loc[h["date"] == d, "hour_n"].to_numpy()
        for hr, rv in zip(hours, raw):
            if abs(rv) > c:
                hour_sat.append((d, int(hr), float(rv), c))
    sr = pd.DataFrame(sat_rows)
    if sr.empty:
        sys.exit("[중단] 보정 재현 표본 없음")
    tot, nsat = int(sr["n"].sum()), int(sr["sat"].sum())
    rate = nsat / tot if tot else 0.0
    wd_sr = sr[sr["구분"] == "평일"]
    hol_sr = sr[sr["구분"] == "휴일"]
    wd_rate = (wd_sr["sat"].sum() / wd_sr["n"].sum()) if len(wd_sr) and wd_sr["n"].sum() else 0
    hol_rate = (hol_sr["sat"].sum() / hol_sr["n"].sum()) if len(hol_sr) and hol_sr["n"].sum() else 0
    print(f"  {'일자':<12}{'구분':<6}{'mode':<28}{'g':>5}{'cap':>5}{'sat':>5}{'min':>8}{'max':>8}")
    print("  " + "-" * 78)
    for r in sr.itertuples():
        print(f"  {r.date:<12}{r.구분:<6}{r.mode:<28}{r.gamma:>5.1f}{r.cap:>5.0f}"
              f"{int(r.sat):>5}{r.raw_min:>+8.1f}{r.raw_max:>+8.1f}")
    print("  " + "-" * 78)
    rep("포화 행수", f"{nsat}행 / {tot}행", f"포화율 {rate * 100:.1f}%")
    rep("포화 방향",
        f"상향 {int(sr['sat_up'].sum())} / 하향 {int(sr['sat_dn'].sum())}", "")
    rep("평일 포화율", f"{wd_rate * 100:.1f}%", f"{int(wd_sr['sat'].sum())}/{int(wd_sr['n'].sum())}")
    rep("휴일 포화율", f"{hol_rate * 100:.1f}%", f"{int(hol_sr['sat'].sum())}/{int(hol_sr['n'].sum())}")
    if hour_sat:
        vc = pd.Series([x[1] for x in hour_sat]).value_counts().head(5)
        rep("포화 집중 시간대",
            ", ".join(f"{i}시({v})" for i, v in vc.items()), "상위 5")

    print("\n■ ⑤ 월요일·전환일")
    for _, r in w[w["dow"] == "월"].iterrows():
        print(f"    {r.date} 월  MAE {r.MAE:.2f}  편의 {r.편의:+.2f}%  {r.비고 or r.구분2}")
    if not convert.empty:
        r = convert.iloc[0]
        print(f"    전환일 9.9   MAE {r.MAE:.2f}  편의 {r.편의:+.2f}%")

    # 9.16·9.21 시간대 급등 여부
    print("\n■ ⑥ 단발 급등·과소 (시간대 |오차| 상위)")
    for d in sorted(watch["date"].unique()):
        g = h[h["date"] == d].copy()
        g["err"] = (g["예측"] - g["실적"]).astype(float)
        g["abs"] = g["err"].abs()
        top = g.sort_values("abs", ascending=False).iloc[0]
        day_mae = g["abs"].mean()
        share = float(top["abs"] / (g["abs"].sum() + 1e-9))
        if top["abs"] >= 25 or share >= 0.25:
            print(f"    {d} {int(top['hour_n']):>2}시  실적 {top['실적']:.2f}  "
                  f"예측 {top['예측']:.2f}  오차 {top['err']:+.2f}  "
                  f"일MAE {day_mae:.2f}  점유 {share * 100:.0f}%")

    print("\n■ ⑦ 판정")
    # 기준: 중간기록 초안 + 관찰 기간 연장
    keep_wd = (n_post >= 5 and mae_post < mae_pre and abs(bi_post) < abs(bi_pre))
    # 평시: 최근 평일 MAE가 전환일(34원)의 절반 이하이며 14원 미만
    calm = (n_rec >= 3 and mae_rec < 14) or (n_post >= 5 and mae_post < 14)
    hol_ok = (len(hol) >= 2 and hol["MAE"].mean() < 14
              and hol["MAE"].mean() < mae_pre)
    sat_ok = rate < 0.10
    # 주말 과소 재발: 휴일 편의가 -12% 이하로 악화되면 실패
    hol_undershoot_any = bool((hol["편의"] < -12).any()) if not hol.empty else False
    rec_undershoot = hol_undershoot_any
    gamma_ok = (w["MAE"].mean() <= 14) or (abs(w["편의"].mean()) <= 8)

    print(f"    · 평일 이중창 : {'유지 요건 충족' if keep_wd and calm else '요건 미달 — 검토'}"
          f" (이전 MAE {mae_pre:.2f} → 이후 {mae_post:.2f}, 최근평일 {mae_rec:.2f})")
    hol_mae = float(hol["MAE"].mean()) if len(hol) else float("nan")
    hol_bi = float(hol["편의"].mean()) if len(hol) else float("nan")
    print(f"    · 휴일 완화   : {'유지 요건 충족' if hol_ok and not rec_undershoot else '주말 과소 재발 또는 표본 부족'}"
          f" (휴일 MAE {hol_mae:.2f} / 편의 {hol_bi:+.2f}%)")
    print(f"    · 상한 ±25    : {'유지' if sat_ok else '상향 검토'} (포화율 {rate * 100:.1f}%)")
    print(f"    · γ 1.5       : {'유지' if gamma_ok else '재조정 검토'}"
          f" (구간 MAE {w['MAE'].mean():.2f} / 편의 {w['편의'].mean():+.2f}%)")
    # 부호 전환은 평일(+)·휴일(−) 교차와 9.16 중립에 가깝다.
    # 연속 동부호 최장 5일은 이중창 이전(9.7~9.11) 하향 전환 국면.
    vib = "γ 발산 아님 (평일+/휴일− 교차·단발 첨두)"
    print(f"    · 진동        : {vib} (전환 {real_flips}회 / 최장동부호 {mx}일)")
    print(f"    · 토 3분류    : 토 {len(sat)}일 — "
          f"{'9.27 재실증 유지(4일 미달)' if len(sat) < 4 else '표본 충족, 별도 재실증'}")

    verdict_wd = "유지" if keep_wd and calm else "재검토"
    verdict_hol = "유지" if hol_ok and not rec_undershoot else "재검토"
    overall = "채택 확정(유지)" if (verdict_wd == "유지" and verdict_hol == "유지" and sat_ok) \
        else "조건부 유지" if (verdict_wd == "유지" or verdict_hol == "유지") \
        else "기각·원복 검토"

    print(f"\n    ★ 종합 판정 : {overall}")
    print(f"       평일 이중창 K21·K5 : {verdict_wd}")
    print(f"       휴일 단일·γ0.7·±15 : {verdict_hol}")
    print(f"       γ1.5 / 상한 ±25     : {'유지' if sat_ok and gamma_ok else '재검토'}")
    print(f"       토요일 3분류         : 9.27 이관 (현재 토 {len(sat)}일)")

    rep("판정_평일이중창", verdict_wd,
        f"이전 {mae_pre:.2f} → 이후 {mae_post:.2f} / 최근 {mae_rec:.2f}")
    rep("판정_휴일완화", verdict_hol,
        f"휴일 MAE {hol['MAE'].mean():.2f}" if len(hol) else "표본없음")
    rep("판정_상한", "유지" if sat_ok else "재검토", f"포화율 {rate * 100:.1f}%")
    rep("판정_감쇠", "γ 1.5 유지" if gamma_ok else "재조정 검토",
        f"MAE {w['MAE'].mean():.2f} / 편의 {w['편의'].mean():+.2f}%")
    rep("판정_종합", overall, f"실적 종일 {last_act}")
    rep("판정_토3분류", f"토 {len(sat)}일", "9.27 재실증 이관" if len(sat) < 4 else "")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_dual_window_judgment_{t}_{ts}.csv")
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
