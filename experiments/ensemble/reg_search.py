# -*- coding: utf-8 -*-
"""
reg_search.py — 「가격을 먼저 맞히고 경보」 방식 탐색 (2026-11-08, 팀 작업본) — 결과만 남김
==============================================================================
분류(사건 0/1)는 학습 사례가 적어(육지 급락 9일·제주 14일) 배울 재료가 부족함.
→ 모든 날의 「가격 크기」를 회귀로 예측하고, 그 값을 경보 확률로 바꿈.
  · 육지 급락 : 다음 날 육지 11~15시 최저가          (낮을수록 위험)
  · 제주 급락 : 다음 날 제주 하루 최저가             (낮을수록 위험)
  · 육지 급등 : 다음 날 max(실적 − 급등 기준)        (클수록 위험)
회귀 성분 (매달 1일, 그 이전 자료로만): LightGBM 분위 회귀 3종(급락 α 0.1·0.3·0.5 / 급등 α 0.5·0.7·0.9) · LightGBM L2 · ExtraTrees
점수 → 확률 : 매달 1일 그 이전 자료로 로지스틱 보정 (chronos_mix.calibrate)
앙상블 : 회귀 성분 + 기존 후보(현재 모델·warn_search·crash_search·Chronos) · 상위 k 평균
고르기 (학습 구간 2022–2024만, max_search.py 와 같은 규칙): 휴일·평일 기준 따로, 적중일이 현재 − max(1, 5%) 이상 → 헛경보 합계 최소
판단 시각 : 급락·제주 전날 15시, 급등 전날 21시 (현재와 같음)
실행 : python experiments/ensemble/reg_search.py     산출 : outputs/ensemble/reg_search_report.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import chronos_mix as CM
import events as EVM
import max_search as MS
import tune_alerts as TA
import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import EVAL_FROM, ap_score, fp_at_recall, lgb, recency

warnings.filterwarnings("ignore")
DEV = TA.DEV


def target(task, D):
    """회귀 목표 (일 단위)"""
    S = EVM.load_settings()
    A, _, _ = EVM.load_actuals(S)
    ad = A["date"].iloc[::24].to_numpy()
    L = A["land"].to_numpy(float).reshape(-1, 24)
    J = A["jeju"].to_numpy(float).reshape(-1, 24)
    with np.errstate(all="ignore"):
        if task == "crash":
            v = np.nanmin(L[:, 10:15], 1)
        elif task == "jeju":
            v = np.nanmin(J, 1)
        else:
            thr = EVM.baseline(L, S["median_days"]) + S["spike_add_won"]
            v = np.nanmax(L - thr, 1)
    v[np.isinf(v)] = np.nan
    return D["date"].map(pd.Series(v, index=ad)).to_numpy(float)


def rolling_reg(D, cols, t, task):
    alphas = (0.1, 0.3, 0.5) if task != "spike" else (0.5, 0.7, 0.9)
    K = [f"q{int(a * 100)}" for a in alphas] + ["l2", "et"]
    P = {k: np.full(len(D), np.nan) for k in K}
    ok = ~np.isnan(t)
    dnum = pd.to_datetime(D["date"]).to_numpy().astype("datetime64[D]").astype(int)
    for ym in sorted(D.loc[D["date"] >= EVAL_FROM, "date"].str[:7].unique()):
        start = int(np.datetime64(ym + "-01", "D").astype(int))
        tr = ok & (dnum < start)
        te = (D["date"].str[:7] == ym).to_numpy()
        if tr.sum() < 200:
            continue
        w = recency(start - dnum[tr])
        c = [x for x in cols if D.loc[tr, x].notna().any()]
        X, T, yt = D.loc[tr, c], D.loc[te, c], t[tr]
        base = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8, random_state=0)
        for a, k in zip(alphas, K[:3]):
            P[k][te] = lgb.LGBMRegressor(objective="quantile", alpha=a, **base).fit(X, yt, sample_weight=w).predict(T)
        P["l2"][te] = lgb.LGBMRegressor(**base).fit(X, yt, sample_weight=w).predict(T)
        et = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                           ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.6, n_jobs=8, random_state=0))
        P["et"][te] = et.fit(X, yt, extratreesregressor__sample_weight=w).predict(T)
    return P


def main():
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    a, b = SETTINGS["test"]
    report = {}
    for task in ("crash", "spike", "jeju"):
        monthly = task == "jeju"
        D = MS.rich(W.build(task, daily), task)
        dates, y = D["date"].to_numpy(), D["y"].to_numpy()
        t = target(task, D)
        R = rolling_reg(D, MS.rich_cols(D, task), t, task)
        sign = 1.0 if task == "spike" else -1.0                       # 점수: 클수록 위험
        pool = {}
        for k, v in R.items():
            pool[f"reg_{k}"] = CM.calibrate(pd.DataFrame({"date": dates, "s": sign * v}), y)
        for k, (dd, p) in TA.candidates(task).items():
            pool[k] = pd.Series(p, index=np.asarray(dd)).reindex(dates).to_numpy(float)
        for nm, F in CM.features(task, dates).items():
            pool[nm] = CM.calibrate(F, y)
        if task == "crash" and (OUT / "warnings_crash.csv").exists():   # 현재(교체판) 모델
            wc = rd(OUT / "warnings_crash.csv")
            pool["현재 모델(교체판)"] = pd.Series(wc["확률"].to_numpy(float), index=wc["date"]).reindex(dates).to_numpy(float)
        ref = "reg_l2"
        devm = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y) & ~np.isnan(pool[ref])
        P = {k: np.where(np.isnan(v), np.nanmedian(v[devm]), v) for k, v in pool.items() if np.isfinite(np.nanmedian(v[devm]))}
        key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
        ranked = sorted(P, key=lambda k: key(P[k]))
        for k in range(2, 7):
            P[f"평균 상위{k}"] = np.mean([P[c] for c in ranked[:k]], 0)
        regs = [k for k in ranked if k.startswith("reg_")][:3]
        P["회귀 상위3 평균"] = np.mean([P[c] for c in regs], 0)
        short = sorted(P, key=lambda k: key(P[k]))[:24]
        off = np.array([off_map.get(d) == "휴일" for d in dates])
        cd, clv = TA.current(task)
        cy = TA.labels(task, cd)
        cur_dev = TA.day_stats(clv, cy, (cd >= DEV[0]) & (cd <= DEV[1]) & ~np.isnan(cy) & (clv != ""))
        if task == "crash":                                   # 교체 전 모델 기준(warnings_crash.py 와 같은 고정값)
            cur_dev = {"사건일": 9, "경보적중": 8, "주의이상적중": 9, "헛경보_경보": 77, "헛경보_주의": 548}
        slack = max(1, round(0.05 * cur_dev["사건일"]))
        best = None
        for n in short:
            p = P[n]
            oo = MS.options_monthly(dates, p, y, devm, off) if monthly else MS.options_fixed(p, y, devm, off)
            ow = MS.options_monthly(dates, p, y, devm, ~off) if monthly else MS.options_fixed(p, y, devm, ~off)
            for o1, s1 in oo:
                for o2, s2 in ow:
                    ha, hw = s1[0] + s2[0], s1[1] + s2[1]
                    if ha < cur_dev["경보적중"] - slack or hw < cur_dev["주의이상적중"] - slack:
                        continue
                    k_ = (s1[2] + s2[2] + s1[3] + s2[3], -hw, -ha)
                    if best is None or k_ < best[0]:
                        best = (k_, n, o1, o2, (ha, hw, s1[2] + s2[2], s1[3] + s2[3]))
        _, bn, o1, o2, ds = best
        p = P[bn]
        if monthly:
            lv = np.where(off, MS.monthly_levels(dates, p, y, *o1, off), MS.monthly_levels(dates, p, y, *o2, ~off))
        else:
            lv = np.full(len(p), "정상", dtype=object)
            for g, (ta, tw) in ((off, o1), (~off, o2)):
                lv[g & (p >= tw)] = "주의"
                lv[g & (p >= ta)] = "경보"
            lv[np.isnan(pool[ref])] = ""
        lv = np.asarray(lv, dtype=object)
        Dn, Dc = pd.DataFrame({"date": dates, "y": y}), pd.DataFrame({"date": cd, "y": cy})
        res = {"candidate": bn, "holiday_rule": o1, "weekday_rule": o2,
               "dev": dict(zip(["경보적중", "주의이상적중", "헛경보_경보", "헛경보_주의"], ds)), "dev_baseline": cur_dev, "slack": slack,
               "reg_components_dev": {k: {"FP@R80": key(P[k])[0], "AP": round(-key(P[k])[1], 3)} for k in P if k.startswith("reg_")},
               "top_dev": [(k, key(P[k])[0], round(-key(P[k])[1], 3)) for k in sorted(P, key=lambda k: key(P[k]))[:10]],
               "test_current": W.seg_score(task, Dc, clv, a, b, seg, off_map), "test_chosen": W.seg_score(task, Dn, lv, a, b, seg, off_map)}
        report[task] = res
        print(f"\n=== {task} · 고른 후보 {bn} · 휴일 {o1} · 평일 {o2}")
        print("  회귀 성분 학습:", res["reg_components_dev"])
        print("  학습 상위:", res["top_dev"][:6])
        print(f"  학습: 기준 {cur_dev} → 새 {res['dev']} (허용 {slack})")
        for nm in ("test_current", "test_chosen"):
            s = res[nm]
            print(f"  {nm:<13} 구간 {s['구간']} · 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} · "
                  f"헛경보 경보 {s['헛경보_경보일']}일 주의 {s['헛경보_주의일']}일 | 휴일 {s['휴일']} | 평일 {s['평일']}")
    (OUT / "reg_search_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
