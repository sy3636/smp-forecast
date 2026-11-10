# -*- coding: utf-8 -*-
"""
ensemble.py — 성분 모델 결합 · 편의보정 · 80% 예측 범위
==============================================================================
1) 가중 결합 : 매달 1일, 그 이전의 표본 외 예측으로 MAE 최소 가중(음수 없음·합 1, 선형계획) 산출
             → 그 달 예측에 적용. 가중 산출 창(전체 누적 / 최근 N일)은 개발 구간 성적으로 선택
2) 편의보정 : ŷ' = ŷ + γ·clip(직전 K일 같은 일유형 시간대별 잔차 평균, ±상한)
             (scripts/step6_bias_adjust.py 와 같은 식 — K·γ 는 개발 구간에서 격자 탐색)
3) 80% 범위 : 최근 W일 잔차의 10%·90% 분위수(conformal). 정규화형은 Chronos-2 분위수 폭으로
             시간대별 불확실성을 반영. 개발 구간 포함률 75~85% 를 지키는 방법 중 폭이 좁은 것 선택

실행
  python experiments/ensemble/ensemble.py            # 개발 구간(2024~2025)만 — 선택·조정용
  python experiments/ensemble/ensemble.py --final    # 결정 고정 후 시험 구간(2026) 1회 평가
산출
  outputs/ensemble/price_final.pkl (date, hour, y, n1, 성분별, pred, lo, hi)
  outputs/ensemble/price_report.json
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from common import (DEV, FRESH, HOLD, OOS_START, OUT, TEST, interval_metrics, load_hourly, metrics4, period_mask,
                    fmt_metrics)

# 개발 구간 성적으로 고른 설정 — 바꿀 때는 개발 구간 결과로만 판단할 것
#  v1 (tune_dev.py, 2026-10-26)
#   · 후보 9종(LightGBM L1·L2·차분·적층 × 월/주 재학습, LEAR, LSTM, Chronos-2 제로샷·미세조정) 중
#     하나씩 빼기에서 기여가 0 인 성분을 제외 → 3종. 개발 MAE 7.14 (7종 7.15)
#   · 편의보정 : 최적 γ 0.25 도 MAE 개선 0.00원 → 미적용 (Chronos 가 수준 전환을 이미 추종)
#  v2 (tune_dev2.py, 2026-11-02)
#   · 미세조정 Chronos 분위수를 입력으로 쓰는 적층 LightGBM 추가 → 개발 MAE 7.14 → 7.13
#   · 기각 : Chronos 분기 미세조정(단독 7.37 vs 연 1회 7.36), 평일·휴일 따로 가중(±0.01),
#            급락 확률 반영(편의 1.22→0.89% 이지만 MAE 7.13→7.17 이상 악화)
#   · 범위 : 정규화형 W 91일 + ACI η 0.02 — 개발 포함률 80.2%, 월별 75~85% 밖 2개월, 폭 22.8원
CONFIG = {
    "members": ["lgb_stack_ft_weekly", "lgb_stack_weekly", "lgb_l1_weekly", "chr_ft_lora"],
    "weight_window": 180,          # 가중 산출 창(일). None = 누적 전체
    "segment": None,               # 가중 구분 — None · off · off_mid
    "bias": {"K": 21, "gamma": 0.0, "cap": 25.0, "split": True},
    "crash_adjust": {"alpha": 0.0, "floor": 0.0},
    "interval": {"method": "norm", "W": 91, "eta": 0.02, "chronos": "dem_off"},
}


# ------------------------------------------------------------------ 성분 불러오기
def load_chronos(name):
    with open(OUT / f"chronos_{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_members(names):
    h = load_hourly()[["date", "hour", "smp", "off"]].rename(columns={"smp": "y"})
    h["n1"] = h["y"].shift(24)
    for n in names:
        if n.startswith("chr_"):
            c = load_chronos(n[4:])
            q50 = c["q"][:, :, c["q_levels"].index(0.5)]
            p = pd.DataFrame({"date": np.repeat(c["dates"], 24), "hour": np.tile(np.arange(1, 25), len(c["dates"])),
                              n: q50.ravel()})
            h = h.merge(p, on=["date", "hour"], how="left")
        else:
            p = pd.read_pickle(OUT / f"pred_{n}.pkl")[["date", "hour", "p"]].rename(columns={"p": n})
            h = h.merge(p, on=["date", "hour"], how="left")
    return h[h["date"] >= OOS_START].reset_index(drop=True)


# ------------------------------------------------------------------ 1) 가중 결합
def lad_weights(Pm, y, max_rows=3000):
    """min Σ|y − Pm·w|  s.t. w ≥ 0, Σw = 1  (행이 많으면 일정 간격으로 추려 선형계획 크기 제한)"""
    if len(y) > max_rows:
        sel = np.linspace(0, len(y) - 1, max_rows).astype(int)
        Pm, y = Pm[sel], y[sel]
    n, k = Pm.shape
    I = sparse.identity(n, format="csr")
    A = sparse.hstack([sparse.csr_matrix(Pm), I, -I], format="csr")
    A = sparse.vstack([A, sparse.hstack([sparse.csr_matrix(np.ones((1, k))), sparse.csr_matrix((1, 2 * n))])])
    b = np.r_[y, 1.0]
    c = np.r_[np.zeros(k), np.ones(2 * n)]
    r = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return r.x[:k] if r.success else np.full(k, 1 / k)


def segment_of(df, how):
    """가중을 따로 산출할 구분 — None(전체 하나) · off(평일/휴일) · off_mid(평일/휴일 × 한낮/그 외)"""
    if not how:
        return np.zeros(len(df), int)
    seg = df["off"].to_numpy(int)
    if how == "off_mid":
        seg = seg * 2 + df["hour"].between(11, 15).to_numpy(int)
    return seg


def combine(df, members, window, segment=None):
    """매달 1일 그 이전 표본 외 예측으로 가중 산출 → 그 달 결합 예측 (구분별로 따로 산출 가능)"""
    months = sorted(df["date"].str[:7].unique())
    Pm = df[members].to_numpy(float)
    y = df["y"].to_numpy(float)
    seg = segment_of(df, segment)
    pred = np.full(len(df), np.nan)
    W = {}
    d = pd.to_datetime(df["date"])
    for ym in months:
        start = pd.Timestamp(ym + "-01")
        hist = (d < start).to_numpy()
        if window:
            hist = hist & (d >= start - pd.Timedelta(days=window)).to_numpy()
        month = (df["date"].str[:7] == ym).to_numpy()
        for s in np.unique(seg):
            ok = hist & (seg == s) & ~np.isnan(Pm).any(1) & ~np.isnan(y)
            w = lad_weights(Pm[ok], y[ok]) if ok.sum() >= 24 * 30 else np.full(len(members), 1 / len(members))
            if s == 0 or not segment:
                W[ym] = w                                  # 보고용 — 평일(또는 전체) 가중
            cur = month & (seg == s)
            avail = ~np.isnan(Pm[cur])
            ww = np.where(avail, w, 0)
            ww = ww / np.where(ww.sum(1, keepdims=True) > 0, ww.sum(1, keepdims=True), 1)
            p = np.nansum(np.nan_to_num(Pm[cur]) * ww, 1)
            p[avail.sum(1) == 0] = np.nan
            pred[cur] = p
    return pred, pd.DataFrame(W, index=members).T


def crash_adjust(df, col, alpha, floor=0.0):
    """급락 확률 반영 — 휴일 한낮 예측을 ŷ − α·p·w_h·max(ŷ − floor, 0) 로 낮춤
    p   : 21시 급락 경보 앙상블 확률(warnings_model.py, 표본 외)
    w_h : 그 달 1일 이전 급락일 중 h시에 실제 30원 미만이었던 비율 (시간대 형상, 매달 갱신)"""
    pred = df[col].to_numpy(float).copy()
    path = OUT / "warnings_final.pkl"
    if alpha == 0 or not path.exists():
        return pred
    w = pd.read_pickle(path)
    w = w[(w["stage"] == "21") & (w["label"] == "crash")].set_index("date")["p"]
    nd = len(df) // 24
    dates = df["date"].to_numpy()[::24]
    Y = df["y"].to_numpy(float).reshape(nd, 24)
    P = pred.reshape(nd, 24)
    crash_day = np.nanmin(Y[:, 10:15], 1) < 30
    for ym in sorted(set(d[:7] for d in dates)):
        past = crash_day & (dates < ym + "-01")
        prof = (Y[past][:, 10:15] < 30).mean(0) if past.sum() else np.ones(5)
        for i in np.where([d[:7] == ym for d in dates])[0]:
            p = w.get(dates[i], np.nan)
            if np.isnan(p):
                continue
            P[i, 10:15] -= alpha * p * prof * np.maximum(P[i, 10:15] - floor, 0)
    return P.ravel()


# ------------------------------------------------------------------ 2) 편의보정
def bias_correct(df, col, K, gamma, cap, split):
    if gamma == 0:
        return df[col].to_numpy(float)
    nd = len(df) // 24
    R = (df["y"] - df[col]).to_numpy(float).reshape(nd, 24)
    off = df["off"].to_numpy().reshape(nd, 24)[:, 0]
    corr = np.zeros((nd, 24))
    hist = {0: [], 1: [], "all": []}
    for i in range(nd):
        key = int(off[i]) if split else "all"
        idx = hist[key][-K:]
        if len(idx) >= 7:
            corr[i] = gamma * np.clip(np.nanmean(R[idx], 0), -cap, cap)
        if not np.isnan(R[i]).all():
            hist[int(off[i])].append(i); hist["all"].append(i)
    return df[col].to_numpy(float) + np.nan_to_num(corr).ravel()


# ------------------------------------------------------------------ 3) 80% 범위
def conformal(df, col, W, scale=None, alpha=0.2, eta=0.0):
    """직전 W일 잔차(정규화형은 잔차 ÷ 척도)의 a/2·1−a/2 분위수 → 범위
    eta > 0 이면 적응형(ACI) — 하루가 끝날 때마다 a ← a + η(α − 그날 이탈 비율) 로 수준을 옮겨 포함률 유지"""
    a = alpha
    nd = len(df) // 24
    y = df["y"].to_numpy(float).reshape(nd, 24)
    p = df[col].to_numpy(float).reshape(nd, 24)
    s = np.ones((nd, 24)) if scale is None else np.maximum(scale.reshape(nd, 24), 1.0)
    e = (y - p) / s
    off = df["off"].to_numpy().reshape(nd, 24)[:, 0]
    lo = np.full((nd, 24), np.nan); hi = np.full((nd, 24), np.nan)
    hist = {0: [], 1: []}
    blocks = [range(0, 6), range(6, 10), range(10, 16), range(16, 24)]   # 새벽·아침·한낮·저녁
    for i in range(nd):
        idx = hist[int(off[i])][-W:]
        if len(idx) >= 14:
            for bl in blocks:
                bl = list(bl)
                ee = e[np.ix_(idx, bl)].ravel()
                ee = ee[~np.isnan(ee)]
                ql, qh = np.quantile(ee, [a / 2, 1 - a / 2])
                lo[i, bl] = p[i, bl] + ql * s[i, bl]
                hi[i, bl] = p[i, bl] + qh * s[i, bl]
            ok = ~np.isnan(y[i]) & ~np.isnan(lo[i])
            if eta and ok.any():
                miss = ((y[i][ok] < lo[i][ok]) | (y[i][ok] > hi[i][ok])).mean()
                a = float(np.clip(a + eta * (alpha - miss), 0.02, 0.5))
        if not np.isnan(e[i]).all():
            hist[int(off[i])].append(i)
    return lo.ravel(), hi.ravel()


def chronos_scale(df, name):
    c = load_chronos(name)
    lv = c["q_levels"]
    di = {d: i for i, d in enumerate(c["dates"])}
    ii = df["date"].map(di).to_numpy()
    jj = df["hour"].to_numpy() - 1
    return (c["q"][ii, jj, lv.index(0.9)] - c["q"][ii, jj, lv.index(0.1)]) / 2


def chronos_interval(df, col, name):
    """Chronos-2 분위수 폭을 결합 예측 중심으로 옮긴 범위 (보정 없음, 비교용)"""
    c = load_chronos(name)
    lv = c["q_levels"]
    di = {d: i for i, d in enumerate(c["dates"])}
    ii = df["date"].map(di).to_numpy(); jj = df["hour"].to_numpy() - 1
    q = c["q"][ii, jj]
    p = df[col].to_numpy(float)
    return p - (q[:, lv.index(0.5)] - q[:, lv.index(0.1)]), p + (q[:, lv.index(0.9)] - q[:, lv.index(0.5)])


# ------------------------------------------------------------------ 실행
def run(cfg, periods):
    df = load_members(cfg["members"])
    df["ens"], W = combine(df, cfg["members"], cfg["weight_window"], cfg.get("segment"))
    b = cfg["bias"]
    df["pred"] = bias_correct(df, "ens", b["K"], b["gamma"], b["cap"], b["split"])
    ca = cfg.get("crash_adjust") or {}
    df["pred"] = crash_adjust(df.assign(_p=df["pred"]), "_p", ca.get("alpha", 0.0), ca.get("floor", 0.0))
    iv = cfg["interval"]
    scale = chronos_scale(df, iv["chronos"]) if iv["method"] == "norm" else None
    df["lo"], df["hi"] = conformal(df, "pred", iv["W"], scale, eta=iv.get("eta", 0.0))
    # 경보 모델 입력용 — 급락 보정 전 결합 예측의 범위 (보정 ↔ 경보 확률이 서로를 키우는 순환 방지)
    df["lo0"], df["hi0"] = conformal(df, "ens", iv["W"], scale, eta=iv.get("eta", 0.0))
    rep = {}
    for name, per in periods.items():
        m = period_mask(df["date"], per)
        r = {k: metrics4(df.loc[m, "y"], df.loc[m, c], df.loc[m, "n1"]) for k, c in
             [("members_" + c, c) for c in cfg["members"]] + [("ens", "ens"), ("final", "pred")]}
        r["interval"] = interval_metrics(df.loc[m, "y"], df.loc[m, "lo"], df.loc[m, "hi"])
        months = df.loc[m, "date"].str[:7]
        r["interval_by_month"] = {ym: interval_metrics(df.loc[m & (df["date"].str[:7] == ym), "y"],
                                                       df.loc[m & (df["date"].str[:7] == ym), "lo"],
                                                       df.loc[m & (df["date"].str[:7] == ym), "hi"])["cover"]
                                  for ym in sorted(months.unique())}
        r["final_by_month"] = {ym: metrics4(df.loc[m & (months == ym).reindex(df.index, fill_value=False), "y"],
                                            df.loc[m & (months == ym).reindex(df.index, fill_value=False), "pred"],
                                            df.loc[m & (months == ym).reindex(df.index, fill_value=False), "n1"])
                               for ym in sorted(months.unique())}
        rep[name] = r
    return df, W, rep


def show(rep, W, members):
    for name, r in rep.items():
        print(f"\n[{name}]")
        for c in members:
            print(f"  {c:<14} {fmt_metrics(r['members_' + c])}")
        print(f"  {'결합':<14} {fmt_metrics(r['ens'])}")
        print(f"  {'최종(보정 후)':<12} {fmt_metrics(r['final'])}")
        iv = r["interval"]
        cov = list(r["interval_by_month"].values())
        print(f"  80% 범위 포함률 {iv['cover']:.1f}% (월별 {min(cov):.0f}~{max(cov):.0f}%)  평균 폭 {iv['width']:.1f}원  "
              f"하한 아래 {iv['below']:.1f}% · 상한 위 {iv['above']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    periods = {"dev": DEV}
    if args.final:
        periods.update({"test": TEST, "hold": HOLD, "fresh": FRESH})
    df, W, rep = run(CONFIG, periods)
    show(rep, W, CONFIG["members"])
    print("\n월별 가중(최근 6개월):")
    print(W.tail(6).round(3).to_string())
    df.to_pickle(OUT / "price_final.pkl")              # 예측값 자체는 표본 외 — 경보 모델 입력으로 사용
    W.to_csv(OUT / "price_weights.csv", encoding="utf-8-sig")
    if args.final:
        with open(OUT / "price_report.json", "w", encoding="utf-8") as f:
            json.dump({"config": CONFIG, "report": rep}, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
