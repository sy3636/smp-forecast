# -*- coding: utf-8 -*-
"""
warnings_jeju.py — 제주 급락(0원 이하) 경보 (2026-11-08, 팀 작업본)
==============================================================================
육지 급락 경보와 같은 기준
  · 학습·시험 구간 : config/eval_settings.json (학습 2020–2024 · 시험 2025-01-01~2026-10-31)
  · 판단 시각      : 전날 issue_hour.crash 시 (15시) — 육지 15시 단계와 같은 정보만 씀
                     (D일까지 SMP, 자체 수요 추정, 수요 없는 Chronos 예측, 육지 15시 급락 확률, 달력)
  · 성분 재학습    : 매달 1일 그 이전 자료로만 (EVAL_FROM 부터 표본 외 확률)
  · 비율 고르기    : 학습 구간 「재현율 80%일 때 헛경보 수」 최소, 동률이면 AP 큰 것 (0.1 격자)
  · 단계           : 경보 = 사례 80%를 잡는 확률 이상, 주의 = 95% — **기준값은 매달 1일, 그 전까지의 모든 사례로 다시 정함**
                     (방법 A, 2026-11-08). 육지는 학습 구간 고정 기준이지만, 제주 0원은 2024년에 처음 생겨 학습 구간 사례가
                     16일뿐이라 고정 기준이면 2025~26년에 매일 경보가 나옴. 미래 자료는 쓰지 않음.
                     ※ 이 방식은 고정 기준의 시험 결과를 본 뒤 정했으므로 시험 성적은 조금 낙관적일 수 있음.
라벨 : 그날 제주 SMP ≤ 0원인 시간이 하나라도 있으면 1 (events.py 의 제주 구간과 같은 정의, jeju_rule)
후보 : 로지스틱 · LightGBM 분류 · 최근 빈도 · 육지 신호 — 단독과 앙상블을 모두 학습 구간 성적으로만 비교해 고름.
       시험 구간 성적은 고른 뒤 한 번만 계산해 보고서에 남김.

실행 : python experiments/ensemble/warnings_jeju.py      (warnings_model.py --final, warnings_weekday.py 다음)
산출 : outputs/ensemble/warnings_jeju.csv · warnings_jeju_report.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import events as EVM
from common import DEV, OUT, SETTINGS, TEST, rd
from warnings_model import EVAL_FROM, ap_score, counts, daily_table, fp_at_recall, lgb, recency, thr_at_recall

WD = "월화수목금토일"
LOGIT = ["j1_zero_h", "j_cnt7", "j1_min", "pr_mid_min", "pvc_share", "off"]
FREQ = ["j_cnt7", "j_cnt28"]
LAND = ["land_p", "pr_mid_min", "c_q10_mid_min"]
LGBF = ["j1_zero_h", "j1_min", "j1_mid_min", "j_cnt7", "j_cnt28", "j_zh7", "j1_spread", "jd1_mid_min", "jd1_dip",
        "land_p", "pr_mid_min", "c_q10_mid_min", "dip_morn", "pvc_share", "resid_nuc", "smp_prev_mid_min",
        "off", "is_holiday", "dow", "month"]
COMPONENTS = ["logit", "lgbm", "freq", "land"]


def table():
    """육지 15시 일 단위 표 + 제주 변수 + 라벨"""
    D = daily_table("15")
    S = EVM.load_settings()
    A, _, _ = EVM.load_actuals(S)
    A = A[A["date"].isin(set(D["date"]))]
    days = A["date"].iloc[::24].to_numpy()
    assert (days == D["date"].to_numpy()).all()
    J = A["jeju"].to_numpy(float).reshape(-1, 24)
    L = A["land"].to_numpy(float).reshape(-1, 24)
    jd = rd(EVM.DATA / "processed" / "smp_dataset_hourly.csv", usecols=["date", "hour_n", "demand_fcst_jeju"])
    rec = rd(EVM.DATA / "interim" / "api_smp_recent.csv", usecols=["date", "hour_n", "demand_fcst_jeju"])
    jd = pd.concat([jd[jd["demand_fcst_jeju"].notna()], rec]).drop_duplicates(["date", "hour_n"])
    JD = (pd.DataFrame({"date": np.repeat(days, 24), "hour_n": np.tile(np.arange(1, 25), len(days))})
          .merge(jd, on=["date", "hour_n"], how="left")["demand_fcst_jeju"].to_numpy(float).reshape(-1, 24))
    zero = (J <= 0) if S["jeju_rule"] == "le0" else (J == 0)
    zero = np.where(np.isnan(J), np.nan, zero)
    with np.errstate(all="ignore"):
        zh = np.nansum(zero, 1); zh[np.isnan(J).all(1)] = np.nan
        D["jeju0"] = np.where(np.isnan(zh), np.nan, (zh > 0).astype(float))
        D["jeju0_hours"] = zh
        s = lambda a, k=1: pd.Series(a).shift(k)
        D["j1_zero_h"] = s(zh).to_numpy()                               # D-1 0원 시간 수 (육지 15시와 같이 D-1 가격 확보 가정)
        D["j1_min"] = s(np.nanmin(J, 1)).to_numpy()
        D["j1_mid_min"] = s(np.nanmin(J[:, 10:15], 1)).to_numpy()
        D["j_cnt7"] = s((zh > 0).astype(float)).rolling(7, min_periods=5).sum().to_numpy()
        D["j_cnt28"] = s((zh > 0).astype(float)).rolling(28, min_periods=20).sum().to_numpy()
        D["j_zh7"] = s(zh).rolling(7, min_periods=5).sum().to_numpy()
        D["j1_spread"] = s(np.nanmean(L - J, 1)).to_numpy()
        D["jd1_mid_min"] = s(np.nanmin(JD[:, 10:15], 1)).to_numpy()
        D["jd1_dip"] = s(np.nanmin(JD[:, 10:15], 1) / np.nanmean(JD[:, 7:9], 1)).to_numpy()
    # 육지 15시 급락 확률 (휴일 = warnings_final, 평일 = warnings_weekday — 둘 다 표본 외)
    w = pd.read_pickle(OUT / "warnings_final.pkl")
    hp = w[(w["label"] == "crash") & (w["stage"] == "15")].set_index("date")["p"]
    wk = rd(OUT / "warnings_weekday.csv").set_index("date")["확률"]
    D["land_p"] = D["date"].map(pd.concat([hp, wk]).groupby(level=0).first())
    return D.replace([np.inf, -np.inf], np.nan)


def fit_logit(X, y, w):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    m.fit(X, y, logisticregression__sample_weight=w)
    return m


def rolling(D):
    """매달 1일 그 이전 자료로만 네 성분 학습 → 그 달 확률 (표본 외)"""
    y = D["jeju0"].to_numpy()
    usable = ~np.isnan(y)
    P = {k: np.full(len(D), np.nan) for k in COMPONENTS}
    dnum = pd.to_datetime(D["date"]).to_numpy().astype("datetime64[D]").astype(int)
    for ym in sorted(D.loc[D["date"] >= EVAL_FROM, "date"].str[:7].unique()):
        start = int(np.datetime64(ym + "-01", "D").astype(int))
        tr = usable & (dnum < start)
        te = (D["date"].str[:7] == ym).to_numpy()
        ytr = y[tr].astype(int)
        if ytr.sum() < 1:                              # 육지와 같이 사례 1일 이상부터
            continue
        w = recency(start - dnum[tr])
        P["logit"][te] = fit_logit(D.loc[tr, LOGIT], ytr, w).predict_proba(D.loc[te, LOGIT])[:, 1]
        P["freq"][te] = fit_logit(D.loc[tr, FREQ], ytr, w).predict_proba(D.loc[te, FREQ])[:, 1]
        P["land"][te] = fit_logit(D.loc[tr, LAND], ytr, w).predict_proba(D.loc[te, LAND])[:, 1]
        pos = ytr.sum()
        if pos >= 5:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   scale_pos_weight=(len(ytr) - pos) / pos, verbose=-1, n_jobs=8, random_state=0)
            m.fit(D.loc[tr, LGBF], ytr, sample_weight=w)
            P["lgbm"][te] = m.predict_proba(D.loc[te, LGBF])[:, 1]
        else:
            P["lgbm"][te] = P["logit"][te]
    return P


def grid(n, step=10):
    for c in itertools.product(range(step + 1), repeat=n):
        if sum(c) == step:
            yield tuple(x / step for x in c)


def stats(p, y, th_a, th_w):
    ev = int(y.sum())
    return {"days": int(len(y)), "events": ev, "fp80": fp_at_recall(p, y) if ev else None, "ap": ap_score(p, y) if ev else None,
            "alarm": counts(p, y, th_a), "watch_or_alarm": counts(p, y, th_w)}


def main():
    D = table()
    P = rolling(D)
    y = D["jeju0"].to_numpy()
    base = ~np.isnan(y) & ~np.isnan(P["logit"])
    dev = base & (D["date"] >= DEV[0]).to_numpy() & (D["date"] < DEV[1]).to_numpy()
    test = base & (D["date"] >= TEST[0]).to_numpy() & (D["date"] < TEST[1]).to_numpy()
    Pf = {k: np.nan_to_num(v, nan=np.nanmedian(v)) for k, v in P.items()}

    # 후보 비교 — 학습 구간 성적만으로 (단독 4개 + 2·3·4개 앙상블, 0.1 격자)
    cands = []
    for r in range(1, len(COMPONENTS) + 1):
        for combo in itertools.combinations(COMPONENTS, r):
            for wts in (grid(len(combo)) if r > 1 else [(1.0,)]):
                if r > 1 and min(wts) == 0:
                    continue
                pe = sum(wi * Pf[k] for wi, k in zip(wts, combo))
                key = (fp_at_recall(pe[dev], y[dev]), -ap_score(pe[dev], y[dev]))
                cands.append((key, combo, wts))
    cands.sort(key=lambda c: c[0])
    best_key, best_combo, best_w = cands[0]
    pe = sum(wi * Pf[k] for wi, k in zip(best_w, best_combo))
    th_a, th_w = thr_at_recall(pe[dev], y[dev], 0.80), thr_at_recall(pe[dev], y[dev], 0.95)   # 고정 기준(참고용)

    # 방법 A — 매달 1일, 그 전까지 표본 외 확률·실제로 기준값을 다시 정함
    TA, TW = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for ym in sorted(D.loc[~np.isnan(P["logit"]), "date"].str[:7].unique()):
        past = base & (D["date"] < ym + "-01").to_numpy()
        if y[past].sum() < 1:
            continue
        m = (D["date"].str[:7] == ym).to_numpy()
        TA[m], TW[m] = thr_at_recall(pe[past], y[past], 0.80), thr_at_recall(pe[past], y[past], 0.95)

    # 보고: 후보별 학습 성적 (단독 + 크기별 최고 앙상블)
    table_rows = []
    for k in COMPONENTS:
        table_rows.append({"후보": k, "비율": {k: 1.0}, "학습_FP@R80": fp_at_recall(Pf[k][dev], y[dev]),
                           "학습_AP": round(ap_score(Pf[k][dev], y[dev]), 3)})
    for r in range(2, len(COMPONENTS) + 1):
        c = next(c for c in cands if len(c[1]) == r)
        table_rows.append({"후보": f"앙상블 {r}개", "비율": dict(zip(c[1], c[2])), "학습_FP@R80": c[0][0], "학습_AP": round(-c[0][1], 3)})
    level = np.where(np.isnan(TA), "", np.where(pe >= TA, "경보", np.where(pe >= TW, "주의", "정상")))

    def lv_stats(m):
        yy, lv = y[m], level[m]
        a, wa = lv == "경보", np.isin(lv, ["경보", "주의"])
        c = lambda on: {"tp": int((on & (yy == 1)).sum()), "fp": int((on & (yy == 0)).sum()), "fn": int((~on & (yy == 1)).sum())}
        return {"days": int(m.sum()), "events": int(yy.sum()), "alarm": c(a), "watch_or_alarm": c(wa)}
    has = level != ""
    rep = {"chosen": {"components": list(best_combo), "weights": dict(zip(best_combo, best_w))},
           "threshold_mode": "monthly", "th_alarm_fixed": float(th_a), "th_watch_fixed": float(th_w),
           "candidates_dev": table_rows,
           "dev": lv_stats(dev & has), "test": lv_stats(test & has),
           "fixed_reference": {"dev": stats(pe[dev], y[dev], th_a, th_w), "test": stats(pe[test], y[test], th_a, th_w)},
           "rule": "비율 = 학습 구간 FP@R80 최소 → AP 최대 · 단계 기준값 = 매달 그 전까지의 사례 80%(경보)·95%(주의)",
           "note": "매달 기준 갱신 방식은 고정 기준의 시험 결과를 본 뒤 정함 — 시험 성적이 조금 낙관적일 수 있음",
           "issue_hour": SETTINGS["issue_hour"]["crash"], "train": SETTINGS["train"], "test_range": SETTINGS["test"]}
    (OUT / "warnings_jeju_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    ok = ~np.isnan(P["logit"])
    out = D.loc[ok, ["date", "dow", "holiday_name", "jeju0", "jeju0_hours", "j1_zero_h", "j_cnt7"]].copy()
    i = out.index
    out.insert(0, "판단시각", f"{SETTINGS['issue_hour']['crash']}시")
    out.insert(1, "과제", "제주 급락")
    out["요일"] = out["dow"].map(lambda d: WD[int(d)])
    out["확률"], out["단계"] = pe[i], level[i]
    out["경보기준"], out["주의기준"] = TA[i], TW[i]
    for k in COMPONENTS:
        out[f"p_{k}"] = P[k][i]
    out = out.rename(columns={"holiday_name": "휴일명", "jeju0": "실제", "jeju0_hours": "0원시간",
                              "j1_zero_h": "전날0원시간", "j_cnt7": "최근7일0원일"})
    cols = ["판단시각", "과제", "date", "요일", "휴일명", "확률", "단계", "경보기준", "주의기준", "실제", "0원시간", "전날0원시간", "최근7일0원일"] + [f"p_{k}" for k in COMPONENTS]
    out[cols].round(4).to_csv(OUT / "warnings_jeju.csv", index=False, encoding="utf-8-sig")

    print(f"[제주 급락 · 전날 {SETTINGS['issue_hour']['crash']}시] 표본 외 확률 {out['date'].min()} ~ {out['date'].max()}")
    print("후보 (학습 구간 성적으로만 비교):")
    for r in table_rows:
        print(f"  {r['후보']:<10} 비율 {r['비율']}  학습 FP@R80 {r['학습_FP@R80']}  AP {r['학습_AP']}")
    print(f"→ 선택 {rep['chosen']} · 기준값 매달 갱신 (참고: 학습 고정 기준이면 주의 ≥ {th_w:.4f} · 경보 ≥ {th_a:.4f})")
    for nm in ("dev", "test"):
        s, f = rep[nm], rep["fixed_reference"][nm]
        print(f"  {'학습' if nm == 'dev' else '시험'}: 사건 {s['events']}/{s['days']}일 · 경보 {s['alarm']} · 주의 이상 {s['watch_or_alarm']}"
              f"   | 고정 기준이면 경보 {f['alarm']} · 주의 이상 {f['watch_or_alarm']}")


if __name__ == "__main__":
    main()
