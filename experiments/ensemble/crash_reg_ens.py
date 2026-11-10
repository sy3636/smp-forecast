# -*- coding: utf-8 -*-
"""
crash_reg_ens.py — 육지 급락: 「한낮 최저가 회귀 → 경보」 모델 확장·앙상블 탐색 (2026-11-08, 팀 작업본) — 결과만 남김
==============================================================================
목표 : 다음 날 육지 11~15시 최저가(원). 판단 시각 전날 15시. 매달 1일 그 이전 자료로만 재학습(표본 외).
회귀 성분 : LightGBM(중앙값·Huber) · CatBoost(중앙값·RMSE) · XGBoost(절대오차) · HistGB(절대오차) · RandomForest · ExtraTrees
           · Ridge · Huber · KNN · SVR · MLP · Chronos-2 15시 한낮 최저 예측(그대로)
앙상블 : 상위 k 예측 평균(k=2~7) · 전체 중앙값 · 스태킹(매달 과거 표본 외 예측으로 음수 없는 선형 가중) · 상위 k 확률 평균
예측 → 확률 : 매달 1일 그 이전 자료로 로지스틱 보정
고르기 (학습 구간 2022–2024만) : 휴일·평일 기준 따로(max_search 규칙), 적중일 ≥ 교체 전 −1 → 헛경보 합계 최소.
           앙상블만 놓고 고른 1등도 따로 보고(과제 조건: LightGBM보다 좋거나 앙상블).
실행 : python experiments/ensemble/crash_reg_ens.py   산출 : outputs/ensemble/crash_reg_ens_report.json · crash_reg_ens_probs.pkl
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

import chronos_mix as CM
import max_search as MS
import reg_search as RS
import tune_alerts as TA
import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import EVAL_FROM, ap_score, fp_at_recall, lgb, recency

warnings.filterwarnings("ignore")
DEV = TA.DEV
BASELINE = {"사건일": 9, "경보적중": 8, "주의이상적중": 9, "헛경보_경보": 77, "헛경보_주의": 548}   # 교체 전 모델(고정)
imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)


def models():
    gb = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8, random_state=0)
    return {
        "lgb_q50": (lambda: lgb.LGBMRegressor(objective="quantile", alpha=0.5, **gb), "w"),
        "lgb_huber": (lambda: lgb.LGBMRegressor(objective="huber", alpha=15.0, **gb), "w"),
        "cat_q50": (lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.5", depth=5, iterations=400, learning_rate=0.05,
                                              verbose=0, random_seed=0, thread_count=8), "w"),
        "cat_rmse": (lambda: CatBoostRegressor(depth=5, iterations=400, learning_rate=0.05, verbose=0, random_seed=0, thread_count=8), "w"),
        "xgb_mae": (lambda: XGBRegressor(objective="reg:absoluteerror", n_estimators=400, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, n_jobs=8, random_state=0), "w"),
        "hgb_mae": (lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                          max_leaf_nodes=15, min_samples_leaf=20, random_state=0), "w"),
        "rf": (lambda: make_pipeline(imp(), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                                  n_jobs=8, random_state=0)), "pipe"),
        "et": (lambda: make_pipeline(imp(), ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.6,
                                                                n_jobs=8, random_state=0)), "pipe"),
        "ridge": (lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0)), "pipe"),
        "huber": (lambda: make_pipeline(imp(), StandardScaler(), HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=2000)), "pipe"),
        "knn": (lambda: make_pipeline(imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance")), "none"),
        "svr": (lambda: make_pipeline(imp(), StandardScaler(), SVR(C=30.0, epsilon=2.0)), "pipe"),
        "mlp": (lambda: make_pipeline(imp(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=800,
                                                                             early_stopping=True, random_state=0)), "none"),
    }


def fit_pred(kind, m, X, y, w, T):
    if kind == "w":
        return m.fit(X, y, sample_weight=w).predict(T)
    if kind == "pipe":
        return m.fit(X, y, **{m.steps[-1][0] + "__sample_weight": w}).predict(T)
    return m.fit(X, y).predict(T)


def rolling(D, cols, t):
    M = models()
    P = {k: np.full(len(D), np.nan) for k in M}
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
        X, T = D.loc[tr, c], D.loc[te, c]
        for k, (mk, kind) in M.items():
            try:
                P[k][te] = fit_pred(kind, mk(), X, t[tr], w, T)
            except Exception as e:                    # 한 모델 실패가 전체를 멈추지 않게
                print(f"  ! {ym} {k}: {type(e).__name__}")
    return P


def stacking(D, R, t, keys):
    """매달 과거 표본 외 예측으로 음수 없는 선형 가중 → 그 달 예측"""
    X = np.column_stack([R[k] for k in keys])
    good = ~np.isnan(X).any(1)
    out = np.full(len(D), np.nan)
    for ym in sorted(D.loc[good, "date"].str[:7].unique()):
        tr = good & ~np.isnan(t) & (D["date"] < ym + "-01").to_numpy()
        te = good & (D["date"].str[:7] == ym).to_numpy()
        if tr.sum() < 120 or not te.any():
            continue
        m = LinearRegression(positive=True).fit(X[tr], t[tr])
        out[te] = m.predict(X[te])
    return out


def main():
    t0 = time.time()
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    a, b = SETTINGS["test"]
    D = MS.rich(W.build("crash", daily), "crash")
    dates, y = D["date"].to_numpy(), D["y"].to_numpy()
    t = RS.target("crash", D)
    R = rolling(D, MS.rich_cols(D, "crash"), t)
    R["chronos_mid"] = D["pr_mid_min"].to_numpy(float)                   # Chronos-2 15시 한낮 최저(중앙값) 그대로
    print(f"성분 학습 끝 {time.time() - t0:.0f}초")
    prob = lambda v: CM.calibrate(pd.DataFrame({"date": dates, "s": -v}), y)   # 낮을수록 위험
    devm0 = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y)
    pool = {f"reg_{k}": prob(v) for k, v in R.items()}
    ref = "reg_lgb_q50"
    devm = devm0 & ~np.isnan(pool[ref])
    fill = lambda p: np.where(np.isnan(p), np.nanmedian(p[devm]), p)
    key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
    P = {k: fill(v) for k, v in pool.items() if np.isfinite(np.nanmedian(v[devm]))}
    singles = sorted(P, key=lambda k: key(P[k]))
    mae = {k: float(np.nanmean(np.abs(R[k[4:]][devm] - t[devm]))) for k in singles}
    # 앙상블 (예측 공간 평균·중앙값·스태킹, 확률 평균)
    E = {}
    for k in range(2, 8):
        top = [s[4:] for s in singles[:k]]
        E[f"예측 평균 상위{k}"] = prob(np.nanmean([R[c] for c in top], 0))
        E[f"확률 평균 상위{k}"] = np.mean([P[s] for s in singles[:k]], 0)
    E["예측 중앙값 전체"] = prob(np.nanmedian([R[c] for c in R], 0))
    st_keys = [s[4:] for s in singles[:6]]
    E["스태킹 상위6"] = prob(stacking(D, R, t, st_keys))
    for k, v in E.items():
        if np.isfinite(np.nanmedian(v[devm])):
            P[k] = fill(v)
    off = np.array([off_map.get(d) == "휴일" for d in dates])
    slack = max(1, round(0.05 * BASELINE["사건일"]))

    def choose(names):
        best = None
        for n in names:
            p = P[n]
            for o1, s1 in MS.options_fixed(p, y, devm, off):
                for o2, s2 in MS.options_fixed(p, y, devm, ~off):
                    ha, hw = s1[0] + s2[0], s1[1] + s2[1]
                    if ha < BASELINE["경보적중"] - slack or hw < BASELINE["주의이상적중"] - slack:
                        continue
                    k_ = (s1[2] + s2[2] + s1[3] + s2[3], -hw, -ha, -ap_score(p[devm], y[devm]))
                    if best is None or k_ < best[0]:
                        best = (k_, n, o1, o2, (ha, hw, s1[2] + s2[2], s1[3] + s2[3]))
        return best

    def levels(n, o1, o2):
        p = P[n]
        lv = np.full(len(p), "정상", dtype=object)
        for g, (ta, tw) in ((off, o1), (~off, o2)):
            lv[g & (p >= tw)] = "주의"
            lv[g & (p >= ta)] = "경보"
        lv[np.isnan(pool[ref])] = ""
        return lv

    table = []
    for n in P:
        bb = choose([n])
        table.append({"후보": n, "앙상블": n in E, "학습_FP@R80": key(P[n])[0], "학습_AP": round(-key(P[n])[1], 3),
                      "학습_헛경보(규칙)": None if bb is None else bb[0][0],
                      "학습_한낮최저_MAE": round(mae[n], 2) if n in mae else None})
    table.sort(key=lambda r: (r["학습_헛경보(규칙)"] if r["학습_헛경보(규칙)"] is not None else 9999, r["학습_FP@R80"]))
    overall, ens_best = choose(list(P)), choose(list(E))
    Dn = pd.DataFrame({"date": dates, "y": y})
    res = {"baseline_dev": BASELINE, "table_dev": table}
    for tag, bb in (("overall", overall), ("ensemble", ens_best)):
        _, n, o1, o2, ds = bb
        res[tag] = {"후보": n, "휴일기준": o1, "평일기준": o2, "dev": dict(zip(["경보적중", "주의이상적중", "헛경보_경보", "헛경보_주의"], ds)),
                    "test": W.seg_score("crash", Dn, levels(n, o1, o2), a, b, seg, off_map)}
    cd, clv = TA.current("crash")
    res["test_current"] = W.seg_score("crash", pd.DataFrame({"date": cd, "y": TA.labels("crash", cd)}), clv, a, b, seg, off_map)
    (OUT / "crash_reg_ens_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.to_pickle({"date": dates, "y": y, "P": P, "R": R}, OUT / "crash_reg_ens_probs.pkl")
    print(f"\n완료 {time.time() - t0:.0f}초 · 학습 구간 순위 (헛경보 규칙 → FP@R80)")
    for r in table[:18]:
        print(f"  {'[앙상블] ' if r['앙상블'] else ''}{r['후보']:<24} 헛경보(규칙) {r['학습_헛경보(규칙)']}  FP@R80 {r['학습_FP@R80']}  AP {r['학습_AP']}  MAE {r['학습_한낮최저_MAE']}")
    for tag in ("overall", "ensemble"):
        s = res[tag]["test"]
        print(f"  {tag:<9} {res[tag]['후보']} · 학습 {res[tag]['dev']} → 시험 경보 {s['성공']} 놓침 {s['실패'] + s['예측 없음']} / {s['구간']} · 헛경보 {s['헛경보_경보일']}일")
    s = res["test_current"]
    print(f"  current   시험 경보 {s['성공']} 놓침 {s['실패'] + s['예측 없음']} / {s['구간']} · 헛경보 {s['헛경보_경보일']}일")


if __name__ == "__main__":
    main()
