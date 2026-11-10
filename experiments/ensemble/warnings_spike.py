# -*- coding: utf-8 -*-
"""
warnings_spike.py — 육지 급등 경보 (새 기준으로 학습, 2026-11-08 팀 작업본)
==============================================================================
바꾼 이유 : 기존 급등 경보(warnings_model.py spike25)는 옛 일 단위 기준(일 최고 ≥ 28일 일 최고 중앙값 × 1.25)으로
           학습돼 있었음. 채점은 새 기준(같은 시각 28일 중앙값 + 20원 구간)이라 라벨을 새 기준으로 맞춤.
고른 방법 : warn_search.py 후보 탐색에서 학습 구간(2022–2024) FP@R80 기준으로 고름 —
           「로지스틱(전체 변수, C=0.05) 0.5 · LightGBM(잎 7) 0.3 · RandomForest 0.2」 (학습 FP@R80 141).
           1등이던 「전체 순위 평균」(139)은 순위를 전체 기간으로 매겨 미래 자료가 섞이므로 제외.
라벨 : 그날 육지 급등 구간이 하나라도 있으면 1 (events_daily.csv)
판단 : 전날 21시 (daily_table("21") + 21시 가격 예측의 시간별 급등 기준 초과 변수)
재학습 : 매달 1일, 그 이전 자료로만. 단계 = 학습 구간 사례 80%(경보)·95%(주의)를 잡는 확률 — 학습 구간 고정(육지 규칙)
실행 : python experiments/ensemble/warnings_spike.py   (events.py 다음 · 다시 events.py — events_daily.csv 라벨을 씀)
산출 : outputs/ensemble/warnings_spike.csv · warnings_spike_report.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import EVAL_FROM, ap_score, fp_at_recall, lgb, recency, thr_at_recall

warnings.filterwarnings("ignore")
WEIGHTS = {"logit_b": 0.5, "lgb_b": 0.3, "rf": 0.2}
WD = "월화수목금토일"


def rolling(D, big):
    y = D["y"].to_numpy()
    usable = ~np.isnan(y)
    P = {k: np.full(len(D), np.nan) for k in WEIGHTS}
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
        bc = [c for c in big if D.loc[tr, c].notna().any()]
        Xb, Tb = D.loc[tr, bc], D.loc[te, bc]
        lr = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                           LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
        P["logit_b"][te] = lr.fit(Xb, ytr, logisticregression__sample_weight=w).predict_proba(Tb)[:, 1]
        if pos >= 5:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=30, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=(len(ytr) - pos) / pos,
                                   verbose=-1, n_jobs=8, random_state=0)
            P["lgb_b"][te] = m.fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
            rf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                      n_jobs=8, random_state=0))
            P["rf"][te] = rf.fit(Xb, ytr, randomforestclassifier__sample_weight=w).predict_proba(Tb)[:, 1]
        else:
            P["lgb_b"][te] = P["rf"][te] = P["logit_b"][te]
    return P


def main():
    daily = rd(OUT / "events_daily.csv")
    D = W.build("spike", daily)
    big = W.feature_sets("spike", D)["big"]
    P = rolling(D, big)
    y, base, dev, test = W.masks(D, P["logit_b"])
    Pf = {k: np.where(np.isnan(v), np.nanmedian(v[dev]), v) for k, v in P.items()}
    pe = sum(w * Pf[k] for k, w in WEIGHTS.items())
    pe = np.where(np.isnan(P["logit_b"]), np.nan, pe)
    th_a, th_w = thr_at_recall(pe[dev], y[dev], 0.80), thr_at_recall(pe[dev], y[dev], 0.95)
    level = np.where(np.isnan(pe), "", np.where(pe >= th_a, "경보", np.where(pe >= th_w, "주의", "정상")))
    hour = SETTINGS["issue_hour"]["spike"]
    out = pd.DataFrame({"판단시각": f"{hour}시", "과제": "육지 급등", "date": D["date"], "요일": [WD[int(d)] for d in D["dow"]],
                        "휴일명": D["holiday_name"], "확률": pe, "단계": level, "실제": y, "급등시간": D["y_hours"],
                        **{f"p_{k}": P[k] for k in WEIGHTS}})
    out = out[out["단계"] != ""]
    out.round(4).to_csv(OUT / "warnings_spike.csv", index=False, encoding="utf-8-sig")
    rep = {"weights": WEIGHTS, "th_alarm": float(th_a), "th_watch": float(th_w), "issue_hour": hour,
           "dev": {"days": int(dev.sum()), "events": int(y[dev].sum()), "fp80": fp_at_recall(pe[dev], y[dev]), "ap": ap_score(pe[dev], y[dev])},
           "label": "그날 육지 급등 구간(같은 시각 28일 중앙값 + 20원) 1개 이상"}
    (OUT / "warnings_spike_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[육지 급등 · 전날 {hour}시] {out['date'].min()} ~ {out['date'].max()} · 비율 {WEIGHTS} · 주의 ≥ {th_w:.4f} · 경보 ≥ {th_a:.4f}")
    print(f"  학습 FP@R80 {rep['dev']['fp80']} · AP {rep['dev']['ap']:.3f}")


if __name__ == "__main__":
    main()
