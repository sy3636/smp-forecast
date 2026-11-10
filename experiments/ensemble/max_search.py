# -*- coding: utf-8 -*-
"""
max_search.py — 헛경보 줄이기 최대 탐색 (2026-11-08, 팀 작업본) — 결과만 남기고 대시보드 모델은 바꾸지 않음
==============================================================================
1) 변수 추가(이미 있는 자료): 연휴 구조(CAL_COLS) · 지난주 같은 요일 가격·사건 · 계절(sin/cos) · 어제 하루 최저
2) 성분 추가(매달 1일 그 이전 자료로만): 로지스틱(넓은 변수, L2·L1) · LightGBM(잎 4) · CatBoost(깊이 3, 규제 강) · ExtraTrees
   + 이전 후보 전부(crash_search · warn_search · 현재 모델) + Chronos 보정 성분 + 상위 k 평균
3) 단계 기준을 휴일·평일 따로: 육지 = 학습 구간 확률 분위(50~99%, 「안 냄」 포함)에서 고름, 제주 = 매달 그 전까지 사례의 재현율 목표로
고르는 규칙(학습 구간 2022–2024만): 경보 적중일·주의 이상 적중일이 각각 현재 − max(1, 사건일 5%) 이상 → 헛경보(경보+주의) 합계 최소
시험 구간은 마지막에 고른 것 하나만 계산.
실행 : python experiments/ensemble/max_search.py     산출 : outputs/ensemble/max_search_report.json · max_search_levels.pkl
"""
import json
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import chronos_mix as CM
import tune_alerts as TA
import warn_search as W
from common import CAL_COLS, OUT, SETTINGS, load_hourly, rd, to_matrix
from warnings_model import EVAL_FROM, ap_score, fp_at_recall, lgb, recency, thr_at_recall

warnings.filterwarnings("ignore")
DEV = TA.DEV
QS = [round(q, 3) for q in list(np.arange(0.05, 0.99, 0.02)) + [0.99, 0.995]]


# ------------------------------------------------------------------ 변수
def rich(D, task):
    h = load_hourly()
    hd = h["date"].iloc[::24].to_numpy()
    cal = h.iloc[::24][["date"] + CAL_COLS].set_index("date")
    P = to_matrix(h, "smp")
    s = lambda a, k: pd.Series(a, index=hd).shift(k)
    extra = pd.DataFrame(index=hd)
    with np.errstate(all="ignore"):
        extra["p1_min"] = s(np.nanmin(P, 1), 1)
        extra["p7_mid_min"] = s(np.nanmin(P[:, 10:15], 1), 7)
        extra["p7_max"] = s(np.nanmax(P, 1), 7)
        extra["p1_n40"] = s((P[:, 10:15] < 40).sum(1).astype(float), 1)
    D = D.join(cal, on="date").join(extra, on="date")
    doy = pd.to_datetime(D["date"]).dt.dayofyear.to_numpy()
    D["doy_s"], D["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    yh = pd.Series(np.nan_to_num(D["y_hours"].to_numpy(float)))
    D["ev7_hours"] = yh.shift(7).to_numpy()
    D["ev1_hours"] = yh.shift(1).to_numpy()
    D["off_x_doy"] = D["off"] * D["doy_c"]
    return D.replace([np.inf, -np.inf], np.nan)


def rich_cols(D, task):
    base = W.feature_sets(task, D)["big"]
    add = CAL_COLS + ["p1_min", "p7_mid_min", "p7_max", "p1_n40", "doy_s", "doy_c", "ev7_hours", "ev1_hours", "off_x_doy"]
    return [c for c in dict.fromkeys(base + add) if c in D.columns]


def rolling(D, cols):
    y = D["y"].to_numpy()
    usable = ~np.isnan(y)
    K = ["r_logit", "r_l1", "r_lgb4", "r_cat3", "r_et"]
    P = {k: np.full(len(D), np.nan) for k in K}
    dnum = pd.to_datetime(D["date"]).to_numpy().astype("datetime64[D]").astype(int)
    for ym in sorted(D.loc[D["date"] >= EVAL_FROM, "date"].str[:7].unique()):
        start = int(np.datetime64(ym + "-01", "D").astype(int))
        tr = usable & (dnum < start)
        te = (D["date"].str[:7] == ym).to_numpy()
        ytr = y[tr].astype(int)
        pos = ytr.sum()
        if pos < 1 or pos == len(ytr):
            continue
        w = recency(start - dnum[tr])
        c = [x for x in cols if D.loc[tr, x].notna().any()]
        X, T = D.loc[tr, c], D.loc[te, c]
        lr = lambda C, l1: make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                         LogisticRegression(C=C, l1_ratio=1.0 if l1 else 0.0, solver="saga" if l1 else "lbfgs",
                                                            class_weight="balanced", max_iter=5000))
        P["r_logit"][te] = lr(0.03, False).fit(X, ytr, logisticregression__sample_weight=w).predict_proba(T)[:, 1]
        P["r_l1"][te] = lr(0.05, True).fit(X, ytr, logisticregression__sample_weight=w).predict_proba(T)[:, 1]
        if pos >= 5:
            spw = (len(ytr) - pos) / pos
            P["r_lgb4"][te] = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=4, min_child_samples=40,
                                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                                 scale_pos_weight=spw, verbose=-1, n_jobs=8, random_state=0
                                                 ).fit(X, ytr, sample_weight=w).predict_proba(T)[:, 1]
            P["r_cat3"][te] = CatBoostClassifier(depth=3, iterations=300, learning_rate=0.05, l2_leaf_reg=10,
                                                 auto_class_weights="Balanced", verbose=0, random_seed=0, thread_count=8
                                                 ).fit(X, ytr, sample_weight=w).predict_proba(T)[:, 1]
            et = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               ExtraTreesClassifier(n_estimators=400, min_samples_leaf=8, max_features=0.5,
                                                    class_weight="balanced", n_jobs=8, random_state=0))
            P["r_et"][te] = et.fit(X, ytr, extratreesclassifier__sample_weight=w).predict_proba(T)[:, 1]
        else:
            for k in ("r_lgb4", "r_cat3", "r_et"):
                P[k][te] = P["r_logit"][te]
    return P


# ------------------------------------------------------------------ 휴일·평일 따로 기준
def options_fixed(p, y, devm, g):
    """그룹 g 의 (경보 기준, 주의 기준) 후보 → 학습 구간 (경보적중, 주의이상적중, 헛경보 경보, 헛경보 주의)"""
    m = devm & g
    pp, yy = p[m], y[m]
    if len(pp) == 0:
        return [((np.inf, np.inf), (0, 0, 0, 0))]
    ths = sorted(set(np.quantile(pp, QS).tolist() + [np.inf]))
    out = []
    for i, ta in enumerate(ths):
        for tw in ths[:i + 1]:
            a, wa = pp >= ta, pp >= tw
            out.append(((ta, tw), (int((a & (yy == 1)).sum()), int((wa & (yy == 1)).sum()),
                                   int((a & (yy == 0)).sum()), int((wa & ~a & (yy == 0)).sum()))))
    return pareto(out)


def options_monthly(dates, p, y, devm, g):
    out = []
    for ra in TA.RA + [1.01]:
        for rw in [r for r in TA.RW + [1.01] if r >= ra]:
            lv = monthly_levels(dates, p, y, ra, rw, g)
            m = devm & g & (lv != "")
            l, yy = lv[m], y[m]
            a, wa = l == "경보", np.isin(l, ["경보", "주의"])
            out.append(((ra, rw), (int((a & (yy == 1)).sum()), int((wa & (yy == 1)).sum()),
                                   int((a & (yy == 0)).sum()), int((wa & ~a & (yy == 0)).sum()))))
    return pareto(out)


def monthly_levels(dates, p, y, ra, rw, g):
    dates = pd.Series(dates)
    TA_, TW_ = np.full(len(p), np.nan), np.full(len(p), np.nan)
    base = ~np.isnan(p) & ~np.isnan(y) & g
    for ym in sorted(dates[~np.isnan(p)].str[:7].unique()):
        past = base & (dates < ym + "-01").to_numpy()
        if y[past].sum() < 1:
            continue
        mm = (dates.str[:7] == ym).to_numpy()
        TA_[mm] = thr_at_recall(p[past], y[past], ra) if ra <= 1 else np.inf
        TW_[mm] = thr_at_recall(p[past], y[past], rw) if rw <= 1 else np.inf
    return np.where(np.isnan(TA_) | np.isnan(p), "", np.where(p >= TA_, "경보", np.where(p >= TW_, "주의", "정상")))


def pareto(opts):
    """적중 둘은 클수록, 헛경보 합은 작을수록 — 지배당하는 후보 제거"""
    keep = []
    for o, (ha, hw, fa, fw) in sorted(opts, key=lambda x: x[1][2] + x[1][3]):
        if any(k[1][0] >= ha and k[1][1] >= hw and k[1][2] + k[1][3] <= fa + fw for k in keep):
            continue
        keep.append((o, (ha, hw, fa, fw)))
    return keep


def main():
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    a, b = SETTINGS["test"]
    report, levels_out = {}, {}
    for task in ("crash", "spike", "jeju"):
        monthly = task == "jeju"
        D = rich(W.build(task, daily), task)
        dates = D["date"].to_numpy()
        y = D["y"].to_numpy()
        pool = rolling(D, rich_cols(D, task))
        base = TA.candidates(task)
        for k, (dd, p) in base.items():
            pool[k] = pd.Series(p, index=np.asarray(dd)).reindex(dates).to_numpy(float)
        for nm, F in CM.features(task, dates).items():
            pool[nm] = CM.calibrate(F, y)
        ref = next(iter(base))
        devm = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y) & ~np.isnan(pool[ref])
        fill = lambda p: np.where(np.isnan(p), np.nanmedian(p[devm]), p)
        P = {k: fill(v) for k, v in pool.items() if np.isfinite(np.nanmedian(v[devm]))}
        key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
        ranked = sorted(P, key=lambda k: key(P[k]))
        for k in range(2, 6):
            P[f"평균 상위{k}"] = np.mean([P[c] for c in ranked[:k]], 0)
        short = sorted(P, key=lambda k: key(P[k]))[:20]
        off = np.array([off_map.get(d) == "휴일" for d in dates])
        cd, clv = TA.current(task)
        cy = TA.labels(task, cd)
        cur_dev = TA.day_stats(clv, cy, (cd >= DEV[0]) & (cd <= DEV[1]) & ~np.isnan(cy) & (clv != ""))
        slack = max(1, round(0.05 * cur_dev["사건일"]))
        best = None
        for n in short:
            p = P[n]
            oo = options_monthly(dates, p, y, devm, off) if monthly else options_fixed(p, y, devm, off)
            ow = options_monthly(dates, p, y, devm, ~off) if monthly else options_fixed(p, y, devm, ~off)
            for o1, s1 in oo:
                for o2, s2 in ow:
                    ha, hw = s1[0] + s2[0], s1[1] + s2[1]
                    if ha < cur_dev["경보적중"] - slack or hw < cur_dev["주의이상적중"] - slack:
                        continue
                    fa = s1[2] + s2[2] + s1[3] + s2[3]
                    k_ = (fa, -hw, -ha)
                    if best is None or k_ < best[0]:
                        best = (k_, n, o1, o2, (ha, hw, s1[2] + s2[2], s1[3] + s2[3]))
        _, bn, o1, o2, ds = best
        p = P[bn]
        if monthly:
            lv1, lv2 = monthly_levels(dates, p, y, *o1, off), monthly_levels(dates, p, y, *o2, ~off)
            lv = np.where(off, lv1, lv2)
        else:
            lv = np.full(len(p), "정상", dtype=object)
            for g, (ta, tw) in ((off, o1), (~off, o2)):
                lv[g & (p >= tw)] = "주의"
                lv[g & (p >= ta)] = "경보"
            lv[np.isnan(pool[ref]) if ref in pool else np.zeros(len(p), bool)] = ""
        lv = np.asarray(lv, dtype=object)
        Dn, Dc = pd.DataFrame({"date": dates, "y": y}), pd.DataFrame({"date": cd, "y": cy})
        res = {"candidate": bn, "holiday_rule": o1, "weekday_rule": o2, "dev": dict(zip(["경보적중", "주의이상적중", "헛경보_경보", "헛경보_주의"], ds)),
               "dev_current": cur_dev, "slack": slack,
               "test_current": W.seg_score(task, Dc, clv, a, b, seg, off_map), "test_chosen": W.seg_score(task, Dn, lv, a, b, seg, off_map),
               "rich_components_dev": {k: {"FP@R80": key(P[k])[0], "AP": round(-key(P[k])[1], 3)} for k in P if k.startswith("r_")}}
        report[task] = res
        levels_out[task] = pd.DataFrame({"date": dates, "p": p, "단계": lv})
        print(f"\n=== {task} · 고른 후보 {bn} · 휴일 기준 {o1} · 평일 기준 {o2}")
        print(f"  새 성분 학습 FP@R80: {res['rich_components_dev']}")
        print(f"  학습: 현재 {cur_dev} → 새 {res['dev']} (허용 {slack})")
        for nm in ("test_current", "test_chosen"):
            s = res[nm]
            print(f"  {nm:<13} 구간 {s['구간']} · 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} · "
                  f"헛경보 경보 {s['헛경보_경보일']}일 주의 {s['헛경보_주의일']}일 | 휴일 {s['휴일']} | 평일 {s['평일']}")
    (OUT / "max_search_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.to_pickle(levels_out, OUT / "max_search_levels.pkl")


if __name__ == "__main__":
    main()
