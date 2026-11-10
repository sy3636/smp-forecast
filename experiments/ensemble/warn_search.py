# -*- coding: utf-8 -*-
"""
warn_search.py — 경보 모델 후보 탐색 (육지 급락 · 육지 급등 · 제주 급락), 2026-11-08 팀 작업본
==============================================================================
목적 : 헛경보를 줄이는 모델을 찾기. 결과만 보고서로 남기고, 대시보드 모델은 바꾸지 않음.
원칙 : 고르기는 **학습 구간(표본 외 확률이 있는 2022–2024) 성적만**으로 한다.
       시험 구간(2025-01-01~2026-10-31) 성적은 고른 모델과 현재 모델에 대해서만 계산해 비교용으로 남김.
라벨 : events.py 의 새 정의로 「그날 사건 구간이 하나라도 있는 날」 (events_daily.csv)
       · crash = 육지 급락(11~15시 < 30원, 휴일·평일 모두) · spike = 육지 급등(같은 시각 28일 중앙값+20원) · jeju = 제주 ≤ 0원
판단 시각 : crash·jeju 전날 15시(daily_table("15")), spike 전날 21시(daily_table("21")) — 현재 설정과 같음
성분 (매달 1일, 그 이전 자료로만 재학습 → 그 달 확률)
  logit_s 작은 로지스틱 · logit_b 전체 변수 로지스틱 · lgb_a / lgb_b LightGBM 두 설정 · et ExtraTrees · rf RandomForest
  · hgb HistGradientBoosting · price 가격 신호 1변수 · freq 최근 빈도
앙상블 : 학습 구간 상위 k개 단순 평균(k=2~5) · 상위 3개 0.1 격자 가중 · 전체 순위 평균 · 스태킹(성분 확률 → 로지스틱, 역시 월별 표본 외)
고르는 기준 : 학습 구간 FP@R80 (사건일 80%를 잡을 때 헛경보 날 수) 최소 → 동률이면 AP 큰 것
단계 기준값 : 육지(crash·spike) = 학습 구간 고정(현재와 같음), 제주 = 매달 갱신(방법 A, 현재와 같음)

실행 : python experiments/ensemble/warn_search.py          (~5–10분)
산출 : outputs/ensemble/warn_search_report.json · warn_search_probs.pkl
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import events as EVM
import warnings_jeju as WJ
from common import DEV, OUT, SETTINGS, TEST, rd
from warnings_model import EVAL_FROM, LGB_FEATS, LOGIT_FEATS, ap_score, daily_table, fp_at_recall, lgb, recency, thr_at_recall

COMP = ["logit_s", "logit_b", "lgb_a", "lgb_b", "et", "rf", "hgb", "price", "freq"]


# ------------------------------------------------------------------ 표 만들기
def lag_counts(D, ev, name):
    s = pd.Series(ev.astype(float))
    D[f"{name}_d1"] = s.shift(1).to_numpy()
    D[f"{name}_c7"] = s.shift(1).rolling(7, min_periods=5).sum().to_numpy()
    D[f"{name}_c28"] = s.shift(1).rolling(28, min_periods=20).sum().to_numpy()


def build(task, daily):
    lab = {"crash": "급락구간", "spike": "급등구간", "jeju": "제주0원구간"}[task]
    if task == "jeju":
        D = WJ.table()
    else:
        D = daily_table("15" if task == "crash" else "21")
    dl = daily.set_index("date")
    ok = D["date"].map(dl["판정"]).fillna(False).astype(bool).to_numpy()
    n = D["date"].map(dl[lab]).to_numpy(float)
    D["y"] = np.where(ok, (n > 0).astype(float), np.nan)
    D["y_hours"] = n
    lag_counts(D, np.nan_to_num(n) > 0, "ev")
    if task == "spike":                                  # 새 정의용 가격 신호: 21시 예측의 문턱 초과
        S = EVM.load_settings()
        A, _, _ = EVM.load_actuals(S)
        days = A["date"].iloc[::24].to_numpy()
        thr = EVM.baseline(A["land"].to_numpy(float).reshape(-1, 24), S["median_days"]) + S["spike_add_won"]
        pf = pd.read_pickle(OUT / "price_final.pkl")
        idx = {d: i for i, d in enumerate(days)}
        T = np.full((len(D), 24), np.nan)
        for i, d in enumerate(D["date"]):
            if d in idx:
                T[i] = thr[idx[d]]
        pm = pf.pivot(index="date", columns="hour", values="ens").reindex(D["date"]).to_numpy()
        hi = pf.pivot(index="date", columns="hour", values="hi0").reindex(D["date"]).to_numpy()
        with np.errstate(all="ignore"):
            D["sp_exc_max"] = np.nanmax(pm - T, 1)
            D["sp_exc_cnt"] = np.nansum(pm >= T, 1).astype(float)
            D["sp_exc_mean"] = np.nanmean(pm - T, 1)
            D["sp_hi_exc_max"] = np.nanmax(hi - T, 1)
            D["sp_hi_exc_cnt"] = np.nansum(hi >= T, 1).astype(float)
            D.loc[np.isnan(pm).all(1), ["sp_exc_cnt", "sp_hi_exc_cnt"]] = np.nan
    D = D.replace([np.inf, -np.inf], np.nan)
    return D


def feature_sets(task, D):
    if task == "crash":
        small = LOGIT_FEATS["crash"] + ["off"]
        big = LGB_FEATS["crash"] + ["off", "ev_d1", "ev_c7", "ev_c28"]
        price = ["pr_mid_min", "c_q10_mid_min"]
    elif task == "spike":
        small = ["sp_exc_max", "sp_exc_cnt", "ev_c7", "dem_max_ratio_ref", "off"]
        big = LGB_FEATS["spike"] + ["sp_exc_max", "sp_exc_cnt", "sp_exc_mean", "sp_hi_exc_max", "sp_hi_exc_cnt", "ev_d1", "ev_c7", "ev_c28"]
        price = ["sp_exc_max", "sp_exc_cnt"]
    else:
        small = WJ.LOGIT
        big = WJ.LGBF + ["ev_c28"]
        price = WJ.LAND
    big = [c for c in dict.fromkeys(big) if c in D.columns]
    return {"small": small, "big": big, "price": price, "freq": ["ev_c7", "ev_c28", "ev_d1"]}


# ------------------------------------------------------------------ 성분 학습 (월별 표본 외)
def logit(C=0.3):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def rolling(D, FS):
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
        if pos < 1 or pos == len(ytr):
            continue
        w = recency(start - dnum[tr])
        bigc = [c for c in FS["big"] if D.loc[tr, c].notna().any()]          # 그 달 학습 자료에서 전부 빈 열은 뺌
        Xs, Xb = D.loc[tr, FS["small"]], D.loc[tr, bigc]
        Ts, Tb = D.loc[te, FS["small"]], D.loc[te, bigc]
        P["logit_s"][te] = logit(0.3).fit(Xs, ytr, logisticregression__sample_weight=w).predict_proba(Ts)[:, 1]
        P["logit_b"][te] = logit(0.05).fit(Xb, ytr, logisticregression__sample_weight=w).predict_proba(Tb)[:, 1]
        P["price"][te] = logit(1.0).fit(D.loc[tr, FS["price"]], ytr, logisticregression__sample_weight=w).predict_proba(D.loc[te, FS["price"]])[:, 1]
        P["freq"][te] = logit(1.0).fit(D.loc[tr, FS["freq"]], ytr, logisticregression__sample_weight=w).predict_proba(D.loc[te, FS["freq"]])[:, 1]
        if pos >= 5:
            spw = (len(ytr) - pos) / pos
            for k, kw in [("lgb_a", dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15)),
                          ("lgb_b", dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=30))]:
                m = lgb.LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                       scale_pos_weight=spw, verbose=-1, n_jobs=8, random_state=0, **kw)
                P[k][te] = m.fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
            for k, M in [("et", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=8, random_state=0)),
                         ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=8, random_state=0))]:
                pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), M)
                P[k][te] = pipe.fit(Xb, ytr, **{pipe.steps[-1][0] + "__sample_weight": w}).predict_proba(Tb)[:, 1]
            hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                                 class_weight="balanced", random_state=0)
            P["hgb"][te] = hgb.fit(Xb, ytr, sample_weight=w).predict_proba(Tb)[:, 1]
        else:                                          # 사례가 적으면 나무 모형 대신 로지스틱 (현재 모델과 같은 처리)
            for k in ("lgb_a", "lgb_b", "et", "rf", "hgb"):
                P[k][te] = P["logit_b"][te]
    return P


def stacking(D, P, keys):
    """성분 확률 → 로지스틱, 매달 그 이전의 표본 외 확률로만 학습"""
    y = D["y"].to_numpy()
    X = np.column_stack([P[k] for k in keys])
    okx = ~np.isnan(X).any(1) & ~np.isnan(y)
    out = np.full(len(D), np.nan)
    for ym in sorted(D.loc[~np.isnan(X).any(1), "date"].str[:7].unique()):
        tr = okx & (D["date"] < ym + "-01").to_numpy()
        te = (D["date"].str[:7] == ym).to_numpy() & ~np.isnan(X).any(1)
        if y[tr].sum() < 3 or not te.any():
            continue
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(np.log(np.clip(X[tr], 1e-4, 1)), y[tr].astype(int))
        out[te] = m.predict_proba(np.log(np.clip(X[te], 1e-4, 1)))[:, 1]
    return out


# ------------------------------------------------------------------ 평가
def masks(D, P0):
    y = D["y"].to_numpy()
    base = ~np.isnan(y) & ~np.isnan(P0)
    dev = base & (D["date"] >= DEV[0]).to_numpy() & (D["date"] < DEV[1]).to_numpy()
    test = base & (D["date"] >= TEST[0]).to_numpy() & (D["date"] < TEST[1]).to_numpy()
    return y, base, dev, test


def levels(D, p, y, base, dev, monthly):
    """단계: 육지 = 학습 구간 고정 기준, 제주 = 매달 갱신"""
    if not monthly:
        ta, tw = thr_at_recall(p[dev], y[dev], 0.80), thr_at_recall(p[dev], y[dev], 0.95)
        return np.where(np.isnan(p), "", np.where(p >= ta, "경보", np.where(p >= tw, "주의", "정상")))
    TA, TW = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for ym in sorted(D.loc[~np.isnan(p), "date"].str[:7].unique()):
        past = base & ~np.isnan(p) & (D["date"] < ym + "-01").to_numpy()
        if y[past].sum() < 1:
            continue
        m = (D["date"].str[:7] == ym).to_numpy()
        TA[m], TW[m] = thr_at_recall(p[past], y[past], 0.80), thr_at_recall(p[past], y[past], 0.95)
    return np.where(np.isnan(TA) | np.isnan(p), "", np.where(p >= TA, "경보", np.where(p >= TW, "주의", "정상")))


def seg_score(task, D, lv, a, b, seg, off_map):
    """시간 구간 채점(경보 모델) — 성공/부분/실패 구간 수, 헛경보 경보·주의 날 수 (휴일·평일 합계와 따로)"""
    L = dict(zip(D["date"], lv))
    s = seg[(seg["key"] == task) & (seg["date"] >= a) & (seg["date"] <= b)]
    j = s["date"].map(L).fillna("").map({"경보": "성공", "주의": "부분", "정상": "실패", "": "예측 없음"})
    yd = dict(zip(D["date"], D["y"]))
    dd = D[(D["date"] >= a) & (D["date"] <= b)]
    no_ev = dd["date"].map(yd) == 0
    lvs = dd["date"].map(L)
    out = {"구간": len(s), **{k: int((j == k).sum()) for k in ("성공", "부분", "실패", "예측 없음")},
           "헛경보_경보일": int((no_ev & (lvs == "경보")).sum()), "헛경보_주의일": int((no_ev & (lvs == "주의")).sum())}
    for g in ("휴일", "평일"):
        sg = s["구분"] == g
        dg = dd["date"].map(off_map) == g
        out[g] = {"구간": int(sg.sum()), "성공": int((j[sg] == "성공").sum()), "부분": int((j[sg] == "부분").sum()),
                  "실패": int((j[sg] == "실패").sum()),
                  "헛경보_경보일": int((no_ev & dg & (lvs == "경보")).sum()), "헛경보_주의일": int((no_ev & dg & (lvs == "주의")).sum())}
    return out


def current_levels(task, D):
    S = EVM.load_settings()
    lv = EVM.load_levels(S, D["date"].to_numpy())
    return lv[task]


def main():
    t0 = time.time()
    daily = rd(OUT / "events_daily.csv")
    seg = rd(OUT / "events_segments.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    report, probs = {}, {}
    for task in ("crash", "spike", "jeju"):
        D = build(task, daily)
        FS = feature_sets(task, D)
        P = rolling(D, FS)
        y, base, dev, test = masks(D, P["logit_s"])
        Pf = {k: np.where(np.isnan(v), np.nanmedian(v[dev]) if np.isfinite(np.nanmedian(v[dev])) else 0, v) for k, v in P.items()}
        key = lambda p: (fp_at_recall(p[dev], y[dev]), -ap_score(p[dev], y[dev]))
        C = {k: Pf[k] for k in COMP}
        ranked = sorted(COMP, key=lambda k: key(Pf[k]))
        for k in range(2, 6):
            C[f"평균 상위{k}"] = np.mean([Pf[c] for c in ranked[:k]], 0)
        top3 = ranked[:3]
        best = None
        for wts in itertools.product(range(11), repeat=3):
            if sum(wts) != 10 or min(wts) == 0:
                continue
            p = sum(w / 10 * Pf[c] for w, c in zip(wts, top3))
            kk = key(p)
            if best is None or kk < best[0]:
                best = (kk, wts, p)
        C["가중 상위3 " + "·".join(f"{c}{w / 10:.1f}" for c, w in zip(top3, best[1]))] = best[2]
        rk = lambda v: pd.Series(v).rank(pct=True).to_numpy()
        C["순위 평균 전체"] = np.mean([rk(Pf[c]) for c in COMP], 0)
        st = stacking(D, P, ranked[:5])
        C["스태킹 상위5"] = np.where(np.isnan(st), np.nanmedian(st), st)

        rows = []
        for name, p in C.items():
            rows.append({"후보": name, "학습_FP@R80": key(p)[0], "학습_AP": round(-key(p)[1], 3)})
        rows.sort(key=lambda r: (r["학습_FP@R80"], -r["학습_AP"]))
        chosen = rows[0]["후보"]
        monthly = task == "jeju"
        a, b = SETTINGS["test"]
        lv_new = levels(D, C[chosen], y, base, dev, monthly)
        cur = current_levels(task if task != "crash" else "crash", D)
        cur_dev = {"FP@R80": None}
        res = {"events_dev": int(y[dev].sum()), "days_dev": int(dev.sum()), "events_test": int(y[test].sum()), "days_test": int(test.sum()),
               "candidates_dev": rows, "chosen": chosen, "threshold": "매달 갱신" if monthly else "학습 구간 고정",
               "test_chosen": seg_score(task, D, lv_new, a, b, seg, off_map),
               "test_current": seg_score(task, D, cur, a, b, seg, off_map),
               "dev_chosen": seg_score(task, D, lv_new, "2022-01-01", "2024-12-31", seg, off_map),
               "dev_current": seg_score(task, D, cur, "2022-01-01", "2024-12-31", seg, off_map)}
        report[task] = res
        probs[task] = pd.DataFrame({"date": D["date"], "y": y, **{k: v for k, v in C.items()}})
        print(f"\n=== {task}  학습 사건 {res['events_dev']}/{res['days_dev']}일 · 시험 사건 {res['events_test']}/{res['days_test']}일  ({time.time() - t0:.0f}초)")
        for r in rows:
            print(f"  {r['후보']:<36} 학습 FP@R80 {r['학습_FP@R80']:>4}  AP {r['학습_AP']}")
        print(f"  → 선택: {chosen}")
        for nm in ("dev_current", "dev_chosen", "test_current", "test_chosen"):
            s = res[nm]
            print(f"  {nm:<13} 구간 {s['구간']} · 성공 {s['성공']} 부분 {s['부분']} 실패 {s['실패']} 없음 {s['예측 없음']} · 헛경보 경보 {s['헛경보_경보일']}일 주의 {s['헛경보_주의일']}일")
    (OUT / "warn_search_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.to_pickle(probs, OUT / "warn_search_probs.pkl")
    print(f"\n완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
