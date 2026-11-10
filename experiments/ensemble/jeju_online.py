# -*- coding: utf-8 -*-
"""
jeju_online.py — 제주 급락: 매달 모델·기준 다시 고르기(온라인 선택) 시험 (2026-11-09, 팀 작업본) — 결과만 남김
==============================================================================
배경 : 제주 0원은 2024년에 처음 생겨 2025–26년에 크게 늘었음(학습 구간 사례 14일). 학습 구간에서 한 번 고른 모델·기준은 시험 구간에서 덜 맞음.
방식(미리 정함, 결과를 보고 바꾸지 않음)
  · 후보 : warnings_jeju_v2.py 와 같은 46개 성분·앙상블 확률 (회귀 13종·Chronos·분류 후보·보정)
  · 매달 1일, 직전 365일(라벨 있는 날)만 보고: 각 후보에서 「그 기간 사건일의 80%를 잡는 확률」을 기준으로 두고
    그때 헛경보 날이 가장 적은 후보를 고름 → 그 달은 그 후보·기준으로 「경보」 한 단계 (휴일·평일 공통)
  · 직전 365일 사건일이 5일 미만이면 그 달은 v2 단계를 그대로 씀
  · 그 달 이후 자료는 쓰지 않음
비교 : v2(현재) — 같은 시험 구간 구간 채점(warn_search.seg_score)
실행 : python experiments/ensemble/jeju_online.py   산출 : outputs/ensemble/jeju_online_report.json · jeju_online_levels.csv
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
from warnings_model import ap_score, fp_at_recall, thr_at_recall

warnings.filterwarnings("ignore")
WINDOW, RECALL, MIN_EVENTS = 365, 0.80, 5


def pool_for_jeju(daily):
    D = MS.rich(W.build("jeju", daily), "jeju")
    dates, y = D["date"].to_numpy(), D["y"].to_numpy()
    t = RS.target("jeju", D)
    R = CRE.rolling(D, MS.rich_cols(D, "jeju"), t)
    R.update(REA.chronos_direct("jeju", dates))
    prob = lambda v: CM.calibrate(pd.DataFrame({"date": dates, "s": -v}), y)
    pool = {f"reg_{k}": prob(v) for k, v in R.items()}
    for k, (dd, p) in TA.candidates("jeju").items():
        pool[f"cls_{k}"] = pd.Series(p, index=np.asarray(dd)).reindex(dates).to_numpy(float)
    for nm, F in CM.features("jeju", dates).items():
        pool[f"cls_{nm}"] = CM.calibrate(F, y)
    ref = ~np.isnan(pool["reg_lgb_q50"])
    devm = (dates >= TA.DEV[0]) & (dates <= TA.DEV[1]) & ~np.isnan(y) & ref
    P = {}
    for k, v in pool.items():
        med = np.nanmedian(v[devm]) if devm.any() else np.nan
        if np.isfinite(med):
            P[k] = np.where(np.isnan(v), med, v)
    key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
    regs = sorted([k for k in P if k.startswith("reg_")], key=lambda k: key(P[k]))
    allr = sorted(P, key=lambda k: key(P[k]))
    E = {"회귀 중앙값 전체": prob(np.nanmedian(np.vstack(list(R.values())), 0))}
    for k in range(2, 8):
        E[f"회귀 예측 평균 상위{k}"] = prob(np.nanmean([R[c[4:]] for c in regs[:k]], 0))
        E[f"확률 평균 상위{k}(전체)"] = np.mean([P[c] for c in allr[:k]], 0)
    for k, v in E.items():
        med = np.nanmedian(v[devm])
        if np.isfinite(med):
            P[k] = np.where(np.isnan(v), med, v)
    return dates, y, ref, P


def main():
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    dates, y, ref, P = pool_for_jeju(daily)
    v2 = rd(OUT / "warnings_jeju_v2.csv").set_index("date")["단계"]
    v2lv = pd.Series(dates).map(v2).fillna("").to_numpy(object)
    dser = pd.Series(dates)
    lv = np.full(len(dates), "", dtype=object)
    picks = []
    for ym in sorted(dser[ref].str[:7].unique()):
        start = np.datetime64(ym + "-01")
        win = (pd.to_datetime(dser).to_numpy() < start) & (pd.to_datetime(dser).to_numpy() >= start - np.timedelta64(WINDOW, "D")) \
              & ~np.isnan(y) & ref
        m = (dser.str[:7] == ym).to_numpy() & ref
        if y[win].sum() < MIN_EVENTS:
            lv[m] = v2lv[m]
            picks.append({"월": ym, "후보": "v2(사례 부족)", "기준": None, "창_사건일": int(y[win].sum())})
            continue
        best = None
        for n, p in P.items():
            th = thr_at_recall(p[win], y[win], RECALL)
            fa = int(((p >= th) & win & (y == 0)).sum())
            if best is None or (fa, -ap_score(p[win], y[win])) < best[0]:
                best = ((fa, -ap_score(p[win], y[win])), n, th)
        _, n, th = best
        lv[m] = np.where(P[n][m] >= th, "경보", "정상")
        picks.append({"월": ym, "후보": n, "기준": round(float(th), 4), "창_사건일": int(y[win].sum()), "창_헛경보": best[0][0]})
    Dn = pd.DataFrame({"date": dates, "y": y})
    a, b = SETTINGS["test"]
    res = {"rule": f"매달 직전 {WINDOW}일, 사건일 {int(RECALL * 100)}% 잡는 기준에서 헛경보 최소 후보", "picks": picks,
           "test_online": W.seg_score("jeju", Dn, lv, a, b, seg, off_map),
           "test_v2": W.seg_score("jeju", Dn, v2lv, a, b, seg, off_map),
           "dev2024_online": W.seg_score("jeju", Dn, lv, "2024-01-01", "2024-12-31", seg, off_map),
           "dev2024_v2": W.seg_score("jeju", Dn, v2lv, "2024-01-01", "2024-12-31", seg, off_map)}
    (OUT / "jeju_online_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame({"date": dates, "단계": lv}).to_csv(OUT / "jeju_online_levels.csv", index=False, encoding="utf-8-sig")
    print("월별 선택:")
    for r in picks:
        if r["월"] >= "2024-06":
            print("  ", r)
    for nm in ("dev2024_v2", "dev2024_online", "test_v2", "test_online"):
        s = res[nm]
        print(f"  {nm:<15} 구간 {s['구간']} · 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} · 헛경보 경보 {s['헛경보_경보일']} 주의 {s['헛경보_주의일']}")


if __name__ == "__main__":
    main()
