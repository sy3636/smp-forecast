# -*- coding: utf-8 -*-
"""
crash_search.py — 육지 급락 경보 후보 확장 탐색 (2026-11-08, 팀 작업본)
==============================================================================
warn_search.py 의 육지 급락 부분을 넓힘. 대시보드 모델은 바꾸지 않고 결과만 남김.
원칙 : 고르기는 학습 구간(표본 외 확률이 있는 2022–2024) 성적만. 시험 구간은 고른 모델·현재 모델만 계산.
라벨 : 그날 육지 급락 구간(11~15시 < 30원)이 있으면 1 — 휴일·평일 모두 (events_daily.csv)
판단 : 전날 15시 (daily_table("15") — 자체 수요 추정·수요 없는 Chronos)
성분 (매달 1일, 그 이전 자료로만 재학습)
  로지스틱 3종(작은 L2 · 전체 L2 · 전체 L1) · LightGBM 3종 · CatBoost 2종 · XGBoost · ExtraTrees · RandomForest · HistGB
  · 가격 신호 2종(Chronos 한낮 최저 / 30원 아래 시간 수)
앙상블 : 상위 k개 평균(k=2~7) · 상위 4개 0.1 격자 가중 · 상위 5개 기하 평균 · 「부스팅 5종」 평균(CatBoost·LightGBM·XGBoost·로지스틱·가격)
고르는 기준 : 학습 구간 FP@R80 최소 → AP 최대. 단계 기준값 = 학습 구간 고정(현재 육지 규칙)
실행 : python experiments/ensemble/crash_search.py   산출 : outputs/ensemble/crash_search_report.json · crash_search_probs.pkl
"""
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import EVAL_FROM, LOGIT_FEATS, ap_score, chronos_by_date, fp_at_recall, lgb, recency

warnings.filterwarnings("ignore")
COMP = ["logit_s", "logit_b", "logit_l1", "lgb_a", "lgb_b", "lgb_c", "cat_a", "cat_b", "xgb", "et", "rf", "hgb", "price", "price2"]


def add_features(D):
    q, lv = chronos_by_date("off", D["date"].to_numpy())
    q10, q50 = q[:, 10:15, lv.index(0.1)], q[:, 10:15, lv.index(0.5)]
    with np.errstate(all="ignore"):
        D["c_q10_n30"] = np.where(np.isnan(q10).all(1), np.nan, (q10 < 30).sum(1))
        D["c_q50_n30"] = np.where(np.isnan(q50).all(1), np.nan, (q50 < 30).sum(1))
        D["c_q50_mid_min"] = np.nanmin(q50, 1)
        doy = pd.to_datetime(D["date"]).dt.dayofyear.to_numpy()
        D["doy_s"], D["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        D["off_pvc"] = D["off"] * D["pvc_share"]
    return D.replace([np.inf, -np.inf], np.nan)


def sets(D):
    small = LOGIT_FEATS["crash"] + ["off"]
    big = [c for c in dict.fromkeys(W.feature_sets("crash", D)["big"] + ["c_q10_n30", "c_q50_n30", "c_q50_mid_min",
                                                                          "doy_s", "doy_c", "off_pvc"]) if c in D.columns]
    return small, big, ["pr_mid_min", "c_q10_mid_min"], ["c_q10_n30", "c_q50_n30", "off"]


def lr(C, pen="l2"):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=C, l1_ratio=1.0 if pen == "l1" else 0.0, solver="saga" if pen == "l1" else "lbfgs",
                                            class_weight="balanced", max_iter=5000))


def rolling(D, small, big, price, price2):
    y = D["y"].to_numpy()
    usable = ~np.isnan(y)
    P = {k: np.full(len(D), np.nan) for k in COMP}
    dnum = pd.to_datetime(D["date"]).to_numpy().astype("datetime64[D]").astype(int)
    for ym in sorted(D.loc[D["date"] >= EVAL_FROM, "date"].str[:7].unique()):
        start = int(np.datetime64(ym + "-01", "D").astype(int))
        tr = usable & (dnum < start)
        te = (D["date"].str[:7] == ym).to_numpy()
        ytr = y[tr].astype(int)
        pos = ytr.sum()
        if pos < 1:
            continue
        w = recency(start - dnum[tr])
        bc = [c for c in big if D.loc[tr, c].notna().any()]
        Xb, Tb = D.loc[tr, bc], D.loc[te, bc]
        fitp = lambda m, cols: m.fit(D.loc[tr, cols], ytr, logisticregression__sample_weight=w).predict_proba(D.loc[te, cols])[:, 1]
        P["logit_s"][te] = fitp(lr(0.3), small)
        P["logit_b"][te] = fitp(lr(0.05), bc)
        P["logit_l1"][te] = fitp(lr(0.1, "l1"), bc)
        P["price"][te] = fitp(lr(1.0), price)
        P["price2"][te] = fitp(lr(1.0), price2)
        if pos >= 5:
            spw = (len(ytr) - pos) / pos
            for k, kw in [("lgb_a", dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15)),
                          ("lgb_b", dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=30)),
                          ("lgb_c", dict(n_estimators=150, learning_rate=0.05, num_leaves=4, min_child_samples=40, extra_trees=True))]:
                m = lgb.LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                       scale_pos_weight=spw, verbose=-1, n_jobs=8, random_state=0, **kw)
                P[k][te] = m.fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
            for k, kw in [("cat_a", dict(depth=4, iterations=300, learning_rate=0.05)),
                          ("cat_b", dict(depth=6, iterations=500, learning_rate=0.03, l2_leaf_reg=5))]:
                m = CatBoostClassifier(auto_class_weights="Balanced", verbose=0, random_seed=0, thread_count=8, **kw)
                P[k][te] = m.fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
            m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=spw, n_jobs=8, random_state=0, eval_metric="logloss")
            P["xgb"][te] = m.fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
            for k, M in [("et", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=8, random_state=0)),
                         ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=8, random_state=0))]:
                pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), M)
                P[k][te] = pipe.fit(Xb, ytr, **{pipe.steps[-1][0] + "__sample_weight": w}).predict_proba(Tb)[:, 1]
            P["hgb"][te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                                          class_weight="balanced", random_state=0).fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
        else:
            for k in ("lgb_a", "lgb_b", "lgb_c", "cat_a", "cat_b", "xgb", "et", "rf", "hgb"):
                P[k][te] = P["logit_b"][te]
    return P


def main():
    t0 = time.time()
    daily, seg = rd(OUT / "events_daily.csv"), rd(OUT / "events_segments.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    D = add_features(W.build("crash", daily))
    small, big, price, price2 = sets(D)
    P = rolling(D, small, big, price, price2)
    y, base, dev, test = W.masks(D, P["logit_s"])
    Pf = {k: np.where(np.isnan(v), np.nanmedian(v[dev]), v) for k, v in P.items()}
    key = lambda p: (fp_at_recall(p[dev], y[dev]), -ap_score(p[dev], y[dev]))
    C = dict(Pf)
    ranked = sorted(COMP, key=lambda k: key(Pf[k]))
    for k in range(2, 8):
        C[f"평균 상위{k}"] = np.mean([Pf[c] for c in ranked[:k]], 0)
    C["기하평균 상위5"] = np.exp(np.mean([np.log(np.clip(Pf[c], 1e-6, 1)) for c in ranked[:5]], 0))
    C["부스팅5 평균(cat_a·lgb_a·xgb·logit_s·price)"] = np.mean([Pf[c] for c in ("cat_a", "lgb_a", "xgb", "logit_s", "price")], 0)
    top4, best = ranked[:4], None
    for wts in itertools.product(range(11), repeat=4):
        if sum(wts) != 10 or min(wts) == 0:
            continue
        p = sum(w / 10 * Pf[c] for w, c in zip(wts, top4))
        kk = key(p)
        if best is None or kk < best[0]:
            best = (kk, wts, p)
    C["가중 상위4 " + "·".join(f"{c}{w / 10:.1f}" for c, w in zip(top4, best[1]))] = best[2]
    rows = sorted([{"후보": n, "학습_FP@R80": key(p)[0], "학습_AP": round(-key(p)[1], 3)} for n, p in C.items()],
                  key=lambda r: (r["학습_FP@R80"], -r["학습_AP"]))
    chosen = rows[0]["후보"]
    a, b = SETTINGS["test"]
    lv_new = W.levels(D, C[chosen], y, base, dev, False)
    cur = W.current_levels("crash", D)
    rep = {"events_dev": int(y[dev].sum()), "days_dev": int(dev.sum()), "candidates_dev": rows, "chosen": chosen,
           "dev_current": W.seg_score("crash", D, cur, "2022-01-01", "2024-12-31", seg, off_map),
           "dev_chosen": W.seg_score("crash", D, lv_new, "2022-01-01", "2024-12-31", seg, off_map),
           "test_current": W.seg_score("crash", D, cur, a, b, seg, off_map),
           "test_chosen": W.seg_score("crash", D, lv_new, a, b, seg, off_map)}
    (OUT / "crash_search_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.to_pickle(pd.DataFrame({"date": D["date"], "y": y, **C}), OUT / "crash_search_probs.pkl")
    print(f"=== 육지 급락  학습 사건 {rep['events_dev']}/{rep['days_dev']}일  ({time.time() - t0:.0f}초)")
    for r in rows:
        print(f"  {r['후보']:<44} 학습 FP@R80 {r['학습_FP@R80']:>4}  AP {r['학습_AP']}")
    print(f"  → 선택: {chosen}")
    for nm in ("dev_current", "dev_chosen", "test_current", "test_chosen"):
        s = rep[nm]
        print(f"  {nm:<13} 구간 {s['구간']} · 성공 {s['성공']} 부분 {s['부분']} 실패 {s['실패']} 없음 {s['예측 없음']} · "
              f"헛경보 경보 {s['헛경보_경보일']}일 주의 {s['헛경보_주의일']}일 | 휴일 {s['휴일']} | 평일 {s['평일']}")


if __name__ == "__main__":
    main()
