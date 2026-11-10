# -*- coding: utf-8 -*-
"""
warnings_jeju_v2.py — 제주 급락 경보 v2 「확률 평균 상위3 앙상블」 (2026-11-09, 팀 작업본)
==============================================================================
고른 과정 (reg_ens_all.py, 학습 구간 2022–2024만) : 회귀 13종 + Chronos 직접 예측 + 분류 후보(현재 모델·warn_search) + Chronos 보정
  + 앙상블 15개 = 46개 중, 학습 구간 FP@R80(→AP) 상위 3개의 확률 평균. 경보 한 단계, 기준은 학습 구간 고정(휴일·평일 따로).
  규칙: 적중일 ≥ v1 −1(13일) 안에서 헛경보 합계 최소 → 학습 헛경보 57일 (v1 170일)
시험(2025-01-01~2026-10-31) : 경보 83/108 · 놓침 25 · 헛경보 204일 (v1: 경보 92·주의 15·놓침 1 · 헛경보 경보 216·주의 220일)
재료 : warnings_jeju.py(v1) 확률과 warn_search.py 제주 후보 확률(warn_search_probs.pkl)을 성분으로 씀 → 그 둘을 먼저 만들어 둘 것
라벨 : 그날 제주 SMP ≤ 0원 시간이 있으면 1 · 판단 : 전날 15시 · 성분은 매달 1일 그 이전 자료로만 재학습
실행 : python experiments/ensemble/warnings_jeju_v2.py (~5분; warnings_jeju.py·events.py 다음, 다시 events.py)
산출 : outputs/ensemble/warnings_jeju_v2.csv · warnings_jeju_v2_report.json  (events.py·대시보드는 v2 가 있으면 v2 를 씀)
"""
import json
import warnings

import numpy as np
import pandas as pd

import chronos_mix as CM
import crash_reg_ens as CRE
import max_search as MS
import reg_ens_all as REA
import reg_search as RS
import tune_alerts as TA
import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import ap_score, fp_at_recall

warnings.filterwarnings("ignore")
WD = "월화수목금토일"
BASELINE = {"사건일": 14, "경보적중": 14, "주의이상적중": 14, "헛경보_경보": 164, "헛경보_주의": 6}   # v1 학습 구간 성적(고정)


def main():
    daily = rd(OUT / "events_daily.csv")
    D = MS.rich(W.build("jeju", daily), "jeju")
    dates, y = D["date"].to_numpy(), D["y"].to_numpy()
    t = RS.target("jeju", D)
    R = CRE.rolling(D, MS.rich_cols(D, "jeju"), t)
    R.update(REA.chronos_direct("jeju", dates))
    prob = lambda v: CM.calibrate(pd.DataFrame({"date": dates, "s": -v}), y)          # 최저가가 낮을수록 위험
    pool = {f"reg_{k}": prob(v) for k, v in R.items()}
    for k, (dd, p) in TA.candidates("jeju").items():                                 # v1(「현재 모델」)·warn_search 후보
        pool[f"cls_{k}"] = pd.Series(p, index=np.asarray(dd)).reindex(dates).to_numpy(float)
    for nm, F in CM.features("jeju", dates).items():
        pool[f"cls_{nm}"] = CM.calibrate(F, y)
    ref = "reg_lgb_q50"
    devm = (dates >= TA.DEV[0]) & (dates <= TA.DEV[1]) & ~np.isnan(y) & ~np.isnan(pool[ref])
    fill = lambda p: np.where(np.isnan(p), np.nanmedian(p[devm]), p)
    key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
    P = {k: fill(v) for k, v in pool.items() if np.isfinite(np.nanmedian(v[devm]))}
    top3 = sorted(P, key=lambda k: key(P[k]))[:3]
    p = np.mean([P[c] for c in top3], 0)
    off = np.array([g == "휴일" for g in pd.Series(dates).map(dict(zip(daily["date"], daily["구분"])))])
    slack = max(1, round(0.05 * BASELINE["사건일"]))
    best = None
    for o1, s1 in MS.options_fixed(p, y, devm, off):
        for o2, s2 in MS.options_fixed(p, y, devm, ~off):
            ha, hw = s1[0] + s2[0], s1[1] + s2[1]
            if ha < BASELINE["경보적중"] - slack or hw < BASELINE["주의이상적중"] - slack:
                continue
            k_ = (s1[2] + s2[2] + s1[3] + s2[3], -hw, -ha)
            if best is None or k_ < best[0]:
                best = (k_, o1, o2, (ha, hw, s1[2] + s2[2], s1[3] + s2[3]))
    _, o1, o2, ds = best
    lv = np.full(len(p), "정상", dtype=object)
    for g, (ta, tw) in ((off, o1), (~off, o2)):
        lv[g & (p >= tw)] = "주의"
        lv[g & (p >= ta)] = "경보"
    lv[np.isnan(pool[ref])] = ""
    hour = SETTINGS["issue_hour"]["crash"]
    out = pd.DataFrame({"판단시각": f"{hour}시", "과제": "제주 급락", "date": dates, "요일": [WD[int(d)] for d in D["dow"]],
                        "휴일명": D["holiday_name"], "구분": np.where(off, "휴일", "평일"), "확률": np.where(lv == "", np.nan, p),
                        "단계": lv, "실제": y, **{f"p_{c}": P[c] for c in top3}})
    out = out[out["단계"] != ""]
    out.round(4).to_csv(OUT / "warnings_jeju_v2.csv", index=False, encoding="utf-8-sig")
    rep = {"model": "확률 평균 상위3 앙상블", "members": top3, "holiday_threshold": o1, "weekday_threshold": o2,
           "levels": "경보 한 단계(학습 구간 고정 기준, 휴일·평일 따로)",
           "dev": dict(zip(["경보적중", "주의이상적중", "헛경보_경보", "헛경보_주의"], ds)), "dev_baseline_v1": BASELINE, "issue_hour": hour}
    (OUT / "warnings_jeju_v2_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[제주 급락 v2 · 전날 {hour}시] 성분 {top3} · 휴일 {o1} · 평일 {o2}")
    print(f"  학습: {rep['dev']} (v1 {BASELINE})")


if __name__ == "__main__":
    main()
