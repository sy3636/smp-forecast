# -*- coding: utf-8 -*-
"""
warnings_model.py — 급락·급등 경보 앙상블 (하루 1행, 매달 재학습)
==============================================================================
라벨
  · 급락 : 휴일·주말 11~15시 육지 SMP 최저 < 30원/kWh                    (과제 ①)
  · 급등 : 그날 최고 SMP ≥ 직전 28일 「일 최고 SMP」 중앙값 × (1 + 25%)  (중간 피드백 4장, 전체 날짜)
판단 시각 (중간 피드백 2장 — 2단 운용)
  · 21시(확정) : D+1 수요예측 확보 → 가격 앙상블(ensemble.py) 예측·범위 사용
  · 15시(근사) : D+1 수요예측 없음 → 자체 수요 추정(D 수요예측·전주·달력, LightGBM) + Chronos-2(수요 없는 변형)
성분
  ① 로지스틱 (중앙값 대체 → 표준화 → L2 C=0.3, 균형 가중, 최근 1년 반감 가중)
  ② LightGBM 분류 (얕은 나무, 균형 가중)
  ③ 가격 신호 — 가격 예측에서 나온 한낮 최저(급락)·일 최고 ÷ 기준(급등)을 1변수 로지스틱으로 확률화
  → 확률 가중 평균. 가중·경보 기준은 개발 구간(2024~2025) 성적으로 결정, 2026 시험은 1회 평가
경보 단계 (중간 피드백 「주의 / 경보」)
  · 경보 : 개발 구간에서 급락(급등)일 80% 를 맞히는 확률값 이상
  · 주의 : 개발 구간에서 95% 를 맞히는 확률값 이상 ~ 경보 미만

실행
  python experiments/ensemble/warnings_model.py               # 개발 구간만
  python experiments/ensemble/warnings_model.py --final       # 시험 구간 1회 평가 + 산출 저장
"""
import argparse
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DEV, FRESH, OUT, TEST, load_hourly, load_lightgbm, load_monthly, to_matrix
from features import nuclear_by_month, seasonal_pv

warnings.filterwarnings("ignore")
lgb = load_lightgbm()

CRASH_TH = 30.0
SPIKE_RATE = 0.25
EVAL_FROM = "2022-01-01"           # 표본 외 확률 산출 시작(학습 2년 확보 — 2020~2021, 2026-11-08 변경, 이전 2023-01-01)
GRID = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]

LOGIT_FEATS = {
    "crash": ["dip_morn", "resid_nuc", "off_prev_mid_min", "smp_prev_mid_min", "pvc_share"],
    "spike": ["dem_max_ratio_ref", "p1_max_ref", "dem_eve_max", "lng_chg", "off", "temp_prev_max", "temp_prev_min"],
}
PRICE_FEAT = {"crash": "pr_mid_min", "spike": "pr_max_ref"}
PRICE_SIGN = {"crash": -1, "spike": 1}          # 값이 작을수록 급락, 클수록 급등


# ------------------------------------------------------------------ 15시용 자체 수요 추정
def demand_estimate(h, dates, Dm, off):
    """D+1 수요예측 대신 쓸 추정치 — D일 수요예측(확보)·전주 동요일·달력. 매달 재학습(표본 외)"""
    path = OUT / "dem_est.pkl"
    if path.exists():
        cached = pd.read_pickle(path)
        if len(cached) == len(dates):                  # 자료 기간이 늘면 다시 계산
            return cached
    n = len(dates)
    sh = lambda M, k: np.vstack([np.full((k, M.shape[1]), np.nan), M[:-k]])
    D1, D7, D6 = sh(Dm, 1), sh(Dm, 7), sh(Dm, 6)
    hol = h["is_holiday"].to_numpy().reshape(-1, 24)[:, 0]
    dt = pd.to_datetime(pd.Series(dates))
    rows = lambda v: np.repeat(np.asarray(v, float), 24)
    X = pd.DataFrame({
        "d1": D1.ravel(), "d7": D7.ravel(), "d6": D6.ravel(), "d1_mean": rows(np.nanmean(D1, 1)),
        "d7_mean": rows(np.nanmean(D7, 1)), "hour": np.tile(np.arange(1, 25), n), "dow": rows(dt.dt.dayofweek),
        "off": rows(off), "off_prev": rows(np.r_[0, off[:-1]]), "off7": rows(np.r_[np.zeros(7), off[:-7]]),
        "hol": rows(hol), "month": rows(dt.dt.month), "doy": rows(dt.dt.dayofyear)})
    y = Dm.ravel()
    day = np.repeat(dates, 24)
    est = np.full(len(y), np.nan)
    for ym in sorted(set(d[:7] for d in dates if d >= "2021-07-01")):
        tr = (day < ym + "-01") & ~np.isnan(y)
        te = np.array([d[:7] == ym for d in day])
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=8)
        m.fit(X[tr], y[tr])
        est[te] = m.predict(X[te])
    out = est.reshape(n, 24)
    pd.to_pickle(out, path)
    return out


# ------------------------------------------------------------------ 일 단위 표
def daily_table(stage):
    h = load_hourly()
    dates = h["date"].iloc[::24].to_numpy()
    n = len(dates)
    P, Dm_true, T = to_matrix(h, "smp"), to_matrix(h, "dem"), to_matrix(h, "temp_land")
    off = h["off"].to_numpy().reshape(-1, 24)[:, 0]
    Dm = Dm_true if stage == "21" else demand_estimate(h, dates, Dm_true, off)
    PV = to_matrix(h, "pv")
    yms = np.array([d[:7] for d in dates])
    nuc = nuclear_by_month(sorted(set(yms))).reindex(yms).to_numpy()
    pvc = seasonal_pv(dates, PV)
    mon = load_monthly()
    lng = mon["pm_vc_lng"].reindex(yms).to_numpy()

    with np.errstate(all="ignore"):
        D = pd.DataFrame({"date": dates})
        D["dow"] = pd.to_datetime(D["date"]).dt.dayofweek
        D["month"] = pd.to_datetime(D["date"]).dt.month
        D["dom"] = pd.to_datetime(D["date"]).dt.day
        D["off"] = off
        D["is_holiday"] = h["is_holiday"].to_numpy().reshape(-1, 24)[:, 0]
        D["holiday_name"] = h["holiday_name"].to_numpy().reshape(-1, 24)[:, 0]
        # 라벨
        mid_min = np.nanmin(P[:, 10:15], 1)
        dmax = np.nanmax(P, 1)
        D["mid_min"], D["day_max"] = mid_min, dmax
        D["max_hour"] = np.where(np.isnan(P).all(1), np.nan, np.nanargmax(np.nan_to_num(P, nan=-1e9), 1) + 1)
        D["crash_hours"] = (P[:, 10:15] < CRASH_TH).sum(1)
        ref28 = pd.Series(dmax).shift(1).rolling(28, min_periods=20).median().to_numpy()
        D["ref28"] = ref28
        D["crash"] = np.where(np.isnan(mid_min), np.nan, (mid_min < CRASH_TH).astype(float))
        for r in (0.20, 0.25, 0.30):
            D[f"spike{int(r * 100)}"] = np.where(np.isnan(dmax) | np.isnan(ref28), np.nan,
                                                 (dmax >= ref28 * (1 + r)).astype(float))
        # 수요(21시 = 공식 수요예측, 15시 = 자체 추정)
        D["dem_mean"] = np.nanmean(Dm, 1)
        D["dem_max"] = np.nanmax(Dm, 1)
        D["dem_mid_min"] = np.nanmin(Dm[:, 10:15], 1)
        D["dem_morn"] = np.nanmean(Dm[:, 7:9], 1)
        D["dip_morn"] = D["dem_mid_min"] / D["dem_morn"]
        D["dem_eve_max"] = np.nanmax(Dm[:, 16:22], 1)
        D["nuc"] = nuc
        D["resid_nuc"] = D["dem_mid_min"] - nuc
        pvc_mid = np.nanmean(pvc[:, 10:15], 1)
        D["pvc_share"] = pvc_mid / np.nanmean(Dm[:, 10:15], 1)
        D["net_mid_min"] = D["dem_mid_min"] - pvc_mid
        dmax_prev = pd.Series(np.nanmax(Dm_true, 1)).shift(1).to_numpy()      # D일 수요예측 최대(확보)
        D["dem_max_ratio1"] = D["dem_max"] / dmax_prev
        dem_ref = pd.Series(np.nanmax(Dm_true, 1)).shift(1).rolling(28, min_periods=20).median().to_numpy()
        D["dem_max_ratio_ref"] = D["dem_max"] / dem_ref
        # 가격 이력 (D일까지)
        prev_mid = pd.Series(mid_min).shift(1).to_numpy()
        D["smp_prev_mid_min"] = prev_mid
        last_off = np.full(n, np.nan); lo_ = np.nan
        for i in range(n):
            last_off[i] = lo_
            if off[i] and not np.isnan(mid_min[i]):
                lo_ = mid_min[i]
        D["off_prev_mid_min"] = last_off
        D["p1_max"] = pd.Series(dmax).shift(1).to_numpy()
        D["p1_mean"] = pd.Series(np.nanmean(P, 1)).shift(1).to_numpy()
        D["p1_max_ref"] = D["p1_max"] / ref28
        D["p7_max_ref"] = pd.Series(dmax).shift(7).to_numpy() / ref28
        D["lng_chg"] = np.nan_to_num(lng / np.r_[lng[0], lng[:-1]] - 1)
        D["first3"] = (D["dom"] <= 3).astype(int)
        tp = pd.DataFrame(T).shift(1)
        D["temp_prev_max"] = tp.iloc[:, :20].max(1).to_numpy()       # D일 1~20시 실측(21시 확보분)
        D["temp_prev_min"] = tp.iloc[:, :20].min(1).to_numpy()

    # 가격 예측 신호 (표본 외)
    di = {d: i for i, d in enumerate(dates)}
    if stage == "21":
        pf = pd.read_pickle(OUT / "price_final.pkl")
        # 급락 보정 전 결합 예측·범위 — 보정된 예측을 쓰면 경보 확률과 보정이 서로를 키움
        pm = pf.pivot(index="date", columns="hour", values="ens").reindex(dates).to_numpy()
        lo = pf.pivot(index="date", columns="hour", values="lo0").reindex(dates).to_numpy()
        hi = pf.pivot(index="date", columns="hour", values="hi0").reindex(dates).to_numpy()
        cname = "dem_off"
    else:
        q, lv = chronos_by_date("off", dates)
        pm, lo, hi = (q[:, :, lv.index(x)] for x in (0.5, 0.1, 0.9))
        cname = "off"
    with np.errstate(all="ignore"):
        D["pr_mid_min"] = np.nanmin(pm[:, 10:15], 1)
        D["pr_lo_mid_min"] = np.nanmin(lo[:, 10:15], 1)
        D["pr_max_ref"] = np.nanmax(pm, 1) / ref28
        D["pr_hi_max_ref"] = np.nanmax(hi, 1) / ref28
        q, lv = chronos_by_date(cname, dates)
        D["c_q10_mid_min"] = np.nanmin(q[:, 10:15, lv.index(0.1)], 1)
        D["c_q90_max_ref"] = np.nanmax(q[:, :, lv.index(0.9)], 1) / ref28
    return D.replace([np.inf, -np.inf], np.nan)


def chronos_by_date(name, dates):
    """Chronos 분위수(일수×24×분위)를 대상 날짜 순서로 맞춤 — 자료 기간이 Chronos 산출 기간과 달라도 됨(밖은 NaN)"""
    with open(OUT / f"chronos_{name}.pkl", "rb") as f:
        c = pickle.load(f)
    idx = {d: i for i, d in enumerate(c["dates"])}
    q = np.full((len(dates),) + c["q"].shape[1:], np.nan, dtype=c["q"].dtype)   # 저장 정밀도(float32) 유지
    for i, d in enumerate(dates):
        if d in idx:
            q[i] = c["q"][idx[d]]
    return q, list(c["q_levels"])


LGB_FEATS = {
    "crash": ["dip_morn", "resid_nuc", "off_prev_mid_min", "smp_prev_mid_min", "pvc_share", "net_mid_min",
              "dem_mid_min", "dem_mean", "month", "dow", "is_holiday", "pr_mid_min", "pr_lo_mid_min", "c_q10_mid_min"],
    "spike": ["dem_max", "dem_max_ratio1", "dem_max_ratio_ref", "dem_eve_max", "p1_max_ref", "p7_max_ref",
              "p1_max", "lng_chg", "first3", "off", "dow", "month", "temp_prev_max", "temp_prev_min",
              "pr_max_ref", "pr_hi_max_ref", "c_q90_max_ref"],
}


# ------------------------------------------------------------------ 성분 학습·예측
def recency(days_before):
    return np.power(0.5, days_before / 365.0)


def fit_logit(X, y, w):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    m.fit(X, y, logisticregression__sample_weight=w)
    return m


def rolling_probs(D, task, label):
    """매달 1일 그 이전 자료로만 세 성분 학습 → 그 달 확률"""
    target = D[label].to_numpy()
    sel = (D["off"] == 1).to_numpy() if task == "crash" else np.ones(len(D), bool)
    usable = sel & ~np.isnan(target)
    if task == "crash":
        usable &= D["dip_morn"].notna().to_numpy()
    probs = {k: np.full(len(D), np.nan) for k in ("logit", "lgbm", "price")}
    dnum = pd.to_datetime(D["date"]).to_numpy().astype("datetime64[D]").astype(int)
    for ym in sorted(D.loc[D["date"] >= EVAL_FROM, "date"].str[:7].unique()):
        start = int(np.datetime64(ym + "-01", "D").astype(int))
        tr = usable & (dnum < start)
        te = sel & (D["date"].str[:7] == ym).to_numpy()
        if task == "crash":
            te &= D["dip_morn"].notna().to_numpy()
        ytr = target[tr].astype(int)
        if not te.any() or ytr.sum() < 1:              # crash_model 과 같이 사례 1일 이상이면 학습
            continue
        w = recency(start - dnum[tr])
        lf = LOGIT_FEATS[task]
        probs["logit"][te] = fit_logit(D.loc[tr, lf], ytr, w).predict_proba(D.loc[te, lf])[:, 1]
        lfg = LGB_FEATS[task]
        pos = ytr.sum()
        if pos >= 5:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   scale_pos_weight=(len(ytr) - pos) / pos, verbose=-1, n_jobs=8, random_state=0)
            m.fit(D.loc[tr, lfg], ytr, sample_weight=w)
            probs["lgbm"][te] = m.predict_proba(D.loc[te, lfg])[:, 1]
        else:                                          # 사례가 너무 적으면 나무 모형 대신 로지스틱 확률
            probs["lgbm"][te] = probs["logit"][te]
        pf = PRICE_FEAT[task]
        ok = tr & D[pf].notna().to_numpy()
        if target[ok].sum() >= 1:
            x = PRICE_SIGN[task] * D.loc[ok, [pf]].to_numpy()
            pm = LogisticRegression(C=1.0, class_weight="balanced").fit(x, target[ok].astype(int),
                                                                          sample_weight=recency(start - dnum[ok]))
            xt = PRICE_SIGN[task] * D.loc[te, [pf]].to_numpy()
            xt = np.where(np.isnan(xt), np.nanmedian(x), xt)
            probs["price"][te] = pm.predict_proba(xt)[:, 1]
    return probs


# ------------------------------------------------------------------ 지표
def thr_at_recall(p, y, r):
    o = np.argsort(-p)
    need = int(np.ceil(r * y.sum()))
    tp = np.cumsum(y[o])
    k = int(np.argmax(tp >= need))
    return p[o][k]


def counts(p, y, th):
    a = p >= th
    return {"tp": int((a & (y == 1)).sum()), "fp": int((a & (y == 0)).sum()), "fn": int((~a & (y == 1)).sum())}


def ap_score(p, y):
    o = np.argsort(-p)
    y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def fp_at_recall(p, y, r=0.8):
    return counts(p, y, thr_at_recall(p, y, r))["fp"]


def evaluate(D, probs, task, label, final):
    y = D[label].to_numpy()
    base = (D["date"] >= DEV[0]).to_numpy() & ~np.isnan(y) & ~np.isnan(probs["logit"])
    dev = base & (D["date"] < DEV[1]).to_numpy()
    test = base & (D["date"] >= TEST[0]).to_numpy() & (D["date"] < TEST[1]).to_numpy()
    out = {"members": {}, "grid": None}
    P = {k: np.nan_to_num(v, nan=np.nanmedian(v)) for k, v in probs.items()}
    for k in P:
        out["members"][k] = {"dev_fp80": fp_at_recall(P[k][dev], y[dev]), "dev_ap": ap_score(P[k][dev], y[dev])}
    best = None
    for w in GRID:                                    # 개발 구간 FP@R80 최소, 동률이면 AP 큰 것
        pe = w[0] * P["logit"] + w[1] * P["lgbm"] + w[2] * P["price"]
        key = (fp_at_recall(pe[dev], y[dev]), -ap_score(pe[dev], y[dev]))
        if best is None or key < best[0]:
            best = (key, w)
    w = best[1]
    pe = w[0] * P["logit"] + w[1] * P["lgbm"] + w[2] * P["price"]
    th_alarm = thr_at_recall(pe[dev], y[dev], 0.80)
    th_watch = thr_at_recall(pe[dev], y[dev], 0.95)
    out.update({"weights": dict(zip(["logit", "lgbm", "price"], w)), "th_alarm": float(th_alarm),
                "th_watch": float(th_watch),
                "dev": {"days": int(dev.sum()), "events": int(y[dev].sum()), "fp80": fp_at_recall(pe[dev], y[dev]),
                        "ap": ap_score(pe[dev], y[dev]), "alarm": counts(pe[dev], y[dev], th_alarm),
                        "watch_or_alarm": counts(pe[dev], y[dev], th_watch)}})
    if final:
        fresh = base & (D["date"] >= FRESH[0]).to_numpy() & (D["date"] < FRESH[1]).to_numpy()
        for name, m in [("test", test), ("fresh", fresh)]:
            if not m.any():
                continue
            ev = int(y[m].sum())
            out[name] = {"days": int(m.sum()), "events": ev, "fp80": fp_at_recall(pe[m], y[m]) if ev else None,
                         "ap": ap_score(pe[m], y[m]) if ev else None, "alarm": counts(pe[m], y[m], th_alarm),
                         "watch_or_alarm": counts(pe[m], y[m], th_watch),
                         "members": {k: {"fp80": fp_at_recall(P[k][m], y[m]) if ev else None,
                                         "ap": ap_score(P[k][m], y[m]) if ev else None} for k in P}}
    return out, pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--stages", nargs="*", default=["21", "15"])
    args = ap.parse_args()
    results, saved = {}, []
    for stage in args.stages:
        D = daily_table(stage)
        for task, label in [("crash", "crash"), ("spike", "spike25"), ("spike", "spike20"), ("spike", "spike30")]:
            if stage == "15" and label != "spike25" and task == "spike":
                continue
            probs = rolling_probs(D, task, label)
            res, pe = evaluate(D, probs, task, label, args.final)
            results[f"{stage}시_{label}"] = res
            d, t = res["dev"], res.get("test")
            msg = (f"[{stage}시 {label:<7}] 가중 {res['weights']}  개발: {d['events']}/{d['days']}일 "
                   f"FP@R80 {d['fp80']} AP {d['ap']:.3f}")
            for key, nm in [("test", "시험"), ("fresh", "새 시험")]:
                t = res.get(key)
                if t:
                    msg += (f" | {nm}: {t['events']}/{t['days']}일 FP@R80 {t['fp80']} "
                            f"경보 적중 {t['alarm']['tp']} 헛경보 {t['alarm']['fp']} · 주의이상 적중 "
                            f"{t['watch_or_alarm']['tp']} 헛경보 {t['watch_or_alarm']['fp']}")
            print(msg)
            print("     성분 개발 FP@R80/AP:", {k: (v["dev_fp80"], round(v["dev_ap"], 3)) for k, v in res["members"].items()})
            # 확률은 모두 표본 외 값 — 항상 저장(가격 예측의 급락 보정 입력). 시험 성적은 --final 일 때만 계산
            cols = ["date", "dow", "off", "holiday_name", label, "mid_min", "day_max", "max_hour", "ref28", "first3"]
            s = D[cols].copy()
            s["stage"], s["task"], s["label"] = stage, task, label
            s["p_logit"], s["p_lgbm"], s["p_price"], s["p"] = probs["logit"], probs["lgbm"], probs["price"], pe
            s["level"] = np.where(pe >= res["th_alarm"], "경보", np.where(pe >= res["th_watch"], "주의", "정상"))
            s.loc[np.isnan(probs["logit"]), "level"] = ""
            saved.append(s.rename(columns={label: "y"}))
    pd.concat(saved).to_pickle(OUT / "warnings_final.pkl")
    if args.final:
        with open(OUT / "warnings_report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
