# -*- coding: utf-8 -*-
"""
explain.py — 과제 3 : 예측 근거를 쉬운 문장으로 (LightGBM 기여도 → 6개 묶음 → 상위 3개 문장)
==============================================================================
  · 설명용 모델 : LightGBM 기본형(lgb_l1 — 사람이 읽을 수 있는 72개 변수만 사용), 매달 1일 그 이전 자료로 재학습
  · 기여도 : LightGBM 내장 TreeSHAP(pred_contrib=True, shap 패키지 TreeExplainer 와 같은 계산)
             asinh 공간 기여를 원/kWh 로 환산 (예측 − 기준값을 기여 비율대로 배분 → 합계 정확히 일치)
  · 기준 : 같은 모델의 최근 4주(직전 28일) 예측 평균 — 「최근 4주보다 왜 높/낮은가」를 묶음별로 나눔
  · 앙상블 예측과의 차이(Chronos-2 등 다른 모델 몫)는 「다른 모델 보정」으로 따로 표시
    → 기준 + 6개 묶음 + 다른 모델 보정 = 앙상블 그날 평균 예측
  · 설명만 하고 예측·튜닝에는 쓰지 않음 (시험 구간 튜닝 없음)

실행
  python experiments/ensemble/explain.py
산출
  outputs/ensemble/explain_daily.pkl / .csv — 날짜별 기준·묶음별 기여(원)·근거 문장
"""
import time

import numpy as np
import pandas as pd

from common import OUT, load_lightgbm
from features import CAL_COLS, build, feature_cols
from run_lgbm import PARAMS, cut_dates

lgb = load_lightgbm()

START = "2024-01-01"
BG = 28                                                # 기준 = 직전 28일
GROUPS = ["최근 가격", "전력 수요", "태양광", "요일·휴일", "연료비·원전", "계절·시간"]
OTHER = "다른 모델 보정"
WD = "월화수목금토일"


def group_of(c):
    if c in ("pvc", "pvc_share"):
        return "태양광"
    if c.startswith("dem") or c == "net_dem":
        return "전력 수요"
    if c == "nuc" or c.startswith(("pm_", "idx_")) or c in ("dubai_usd_bbl", "lng_chg_d1", "q60_chg_d1"):
        return "연료비·원전"
    if c.startswith("p"):
        return "최근 가격"
    if c in ("dow", "off", "off_prev", "off_next") or c in CAL_COLS:
        return "요일·휴일"
    if c in ("hour", "month", "dom", "doy_sin", "doy_cos"):
        return "계절·시간"
    raise KeyError(c)


# 묶음별 문장 — (값이 오르는 쪽, 내리는 쪽). 변수 변화 방향과 기여 부호가 맞을 때만 쓰고, 아니면 「○○ 영향으로」
PHRASE = {
    "최근 가격": ("최근 가격이 높아", "최근 가격이 낮아"),
    "전력 수요": ("전력 수요가 많아", "전력 수요가 적어"),
    "태양광": ("태양광 발전이 적어", "태양광 발전이 많아"),
    "요일·휴일": ("평일이라", "휴일·주말이라"),
    "연료비·원전": ("연료비 단가가 올라", "연료비 단가가 내려"),
    "계절·시간": (None, None),
}


def fmt_won(v):
    return f"{'+' if v >= 0 else '−'}{abs(v):.1f}원"


def phrase(g, v, s):
    up, down = PHRASE[g]
    if v > 0 and s > 0 and up:
        return up
    if v < 0 and s < 0 and down:
        return down
    return f"{g} 영향으로"


def won_contrib(m, Xte, med, mad, groups):
    """행별 묶음 기여(원) — asinh 공간 TreeSHAP 을 예측(원) − 기준값(원) 에 비례 배분"""
    c = m.booster_.predict(Xte, pred_contrib=True)
    zb, cf = c[:, -1], c[:, :-1]
    z = zb + cf.sum(1)
    P, Pb = np.sinh(z) * mad + med, np.sinh(zb) * mad + med
    dz = z - zb
    ratio = np.where(np.abs(dz) > 1e-9, (P - Pb) / np.where(np.abs(dz) > 1e-9, dz, 1), mad * np.cosh(zb))
    won = cf * ratio[:, None]
    out = pd.DataFrame({g: won[:, groups == g].sum(1) for g in GROUPS})
    out["pred"] = P
    return out


def main():
    X = build()
    X = X[X["date"] >= "2021-01-08"].reset_index(drop=True)
    cols = feature_cols(X)
    groups = np.array([group_of(c) for c in cols])
    days = pd.to_datetime(pd.Series(X["date"].unique()))
    cuts = [c for c in cut_dates(X["date"], "monthly") if c >= START or c == "9999-12-31"]

    # 문장 방향 판단용 일 단위 변수
    d = X.groupby("date").agg(dem=("dem_mean", "first"), pvc=("pvc", "sum"), p1=("p1_mean", "first"),
                              p28=("pma28_mean", "first"), lng=("pm_vc_lng", "first"), off=("off", "first"))

    rows, t0 = [], time.time()
    for a, b in zip(cuts[:-1], cuts[1:]):
        te_days = X["date"][(X["date"] >= a) & (X["date"] < b)].unique()
        if not len(te_days):
            continue
        tr = (X["date"] < a) & X["y"].notna()
        ytr = X.loc[tr, "y"].to_numpy()
        med = float(np.median(ytr))
        mad = float(np.median(np.abs(ytr - med))) * 1.4826 or 1.0
        m = lgb.LGBMRegressor(objective="l1", **PARAMS).fit(X.loc[tr, cols].to_numpy(float), np.arcsinh((ytr - med) / mad))
        bg0 = (pd.Timestamp(a) - pd.Timedelta(days=BG)).strftime("%Y-%m-%d")
        sel = (X["date"] >= bg0) & (X["date"] < b)
        wc = won_contrib(m, X.loc[sel, cols].to_numpy(float), med, mad, groups)
        daily = wc.groupby(X.loc[sel, "date"].to_numpy()).mean()          # 그날 24시간 평균
        for dd in te_days:
            i = daily.index.get_loc(dd)
            bg = daily.iloc[max(0, i - BG):i]
            r = {"date": dd, "ref": bg["pred"].mean(), "lgb": daily.iloc[i]["pred"]}
            for g in GROUPS:
                r[g] = daily.iloc[i][g] - bg[g].mean()
            rows.append(r)
        print(f"  {a[:7]} {time.time() - t0:5.0f}초", flush=True)

    E = pd.DataFrame(rows).set_index("date")
    p = pd.read_pickle(OUT / "price_final.pkl")
    ens = p.groupby("date")["pred"].mean()
    E["ens"] = ens.reindex(E.index)
    E[OTHER] = E["ens"] - E["lgb"]
    E = E[E["ens"].notna()]

    # 변수 변화 방향 (+ : 가격을 올리는 쪽)
    dd = d.reindex(days.dt.strftime("%Y-%m-%d"))
    roll = lambda s: s.shift(1).rolling(BG, min_periods=14).mean()
    sgn = lambda r, th: np.where(r > 1 + th, 1, np.where(r < 1 - th, -1, 0))
    dirs = pd.DataFrame({
        "최근 가격": sgn(dd["p1"] / dd["p28"], 0.02),
        "전력 수요": sgn(dd["dem"] / roll(dd["dem"]), 0.01),
        "태양광": -sgn(dd["pvc"] / roll(dd["pvc"]), 0.05),
        "요일·휴일": np.where(dd["off"] > 0, -1, 1),
        "연료비·원전": sgn(dd["lng"] / dd["lng"].shift(BG), 0.005),
        "계절·시간": 0,
    }, index=dd.index).reindex(E.index)

    sent, top = [], []
    for dt, r in E.iterrows():
        t = pd.Timestamp(dt)
        diff = r["ens"] - r["ref"]
        order = sorted(GROUPS, key=lambda g: -abs(r[g]))[:3]
        reasons = [f"{phrase(g, r[g], dirs.at[dt, g])} {fmt_won(r[g])}" for g in order]
        lead = (f"{t.month}. {t.day}.({WD[t.dayofweek]}) 평균 {r['ens']:.1f}원 예측 — 최근 4주 평균 수준 {r['ref']:.1f}원보다 "
                f"{abs(diff):.1f}원 {'높음' if diff >= 0 else '낮음'}.")
        tail = f" (Chronos-2 등 다른 모델 보정 {fmt_won(r[OTHER])} 포함)" if abs(r[OTHER]) >= abs(r[order[-1]]) else ""
        sent.append(f"{lead} 주요 근거: {', '.join(reasons)}.{tail}")
        top.append(order)
    E["sentence"] = sent
    E["top3"] = top
    E = E.reset_index()

    E.to_pickle(OUT / "explain_daily.pkl")
    E.drop(columns=["top3"]).round(2).to_csv(OUT / "explain_daily.csv", index=False, encoding="utf-8-sig")
    chk = (E["ref"] + E[GROUPS + [OTHER]].sum(axis=1) - E["ens"]).abs().max()
    print(f"완료 {time.time() - t0:.0f}초 · {len(E)}일 · 합계 검산 최대 오차 {chk:.2e}원")
    t26 = E[E["date"] >= "2025-01-01"]
    print("시험 구간(2025~) 묶음별 평균 |기여|(원):", t26[GROUPS + [OTHER]].abs().mean().round(2).to_dict())
    for s in E.sample(5, random_state=1)["sentence"]:
        print(" ·", s)


if __name__ == "__main__":
    main()
