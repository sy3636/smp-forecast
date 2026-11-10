# -*- coding: utf-8 -*-
"""
reg_ens_all.py — 육지 급등·제주 급락: 회귀 13종 + 분류 후보 + 앙상블, 기준(고정·매달) 함께 고르기 (2026-11-08) — 결과만 남김
==============================================================================
목표(회귀) : 육지 급등 = 다음 날 max(실적 − 급등 기준) (전날 21시) · 제주 급락 = 다음 날 제주 하루 최저가 (전날 15시)
성분 : crash_reg_ens.models() 13종(매달 1일, 그 이전 자료로만) + Chronos-2 직접 예측 + 기존 분류 후보(현재 모델·warn_search) + Chronos 보정
앙상블 : 회귀 중앙값 · 회귀 상위 k 평균 · 회귀 스태킹 · 전체 후보 확률 평균 상위 k(2~7)
기준 : 휴일·평일 따로, 「학습 구간 고정(확률 분위)」과 「매달 갱신(재현율 목표)」 둘 다 후보
고르기 (학습 구간 2022–2024만) : 경보 적중일·주의 이상 적중일 ≥ 현재 − max(1, 5%) → 헛경보(경보+주의) 합계 최소 → AP 큰 것
실행 : python experiments/ensemble/reg_ens_all.py      산출 : outputs/ensemble/reg_ens_all_report.json
"""
import json
import time
import warnings

import numpy as np
import pandas as pd

import chronos_mix as CM
import crash_reg_ens as CRE
import max_search as MS
import reg_search as RS
import tune_alerts as TA
import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import ap_score, fp_at_recall

warnings.filterwarnings("ignore")
DEV = TA.DEV


def chronos_direct(task, dates):
    if task == "spike":
        import events as EVM
        S = EVM.load_settings()
        A, _, _ = EVM.load_actuals(S)
        ad = A["date"].iloc[::24].to_numpy()
        thr = EVM.baseline(A["land"].to_numpy(float).reshape(-1, 24), S["median_days"]) + S["spike_add_won"]
        ti = {d: i for i, d in enumerate(ad)}
        T = np.array([thr[ti[d]] if d in ti else np.full(24, np.nan) for d in dates])
        out = {}
        for nm in ("dem_off", "ft_lora"):
            q, lv = CM.chronos(nm, dates)
            with np.errstate(all="ignore"):
                out[f"chronos_{nm}"] = np.nanmax(q[:, :, list(lv).index(0.5)] - T, 1)
        return out
    q, lv = CM.chronos("off", dates)
    with np.errstate(all="ignore"):
        return {"chronos_off_min": np.nanmin(q[:, :, list(lv).index(0.5)], 1)}


def run(task, daily, seg, off_map):
    t0 = time.time()
    D = MS.rich(W.build(task, daily), task)
    dates, y = D["date"].to_numpy(), D["y"].to_numpy()
    t = RS.target(task, D)
    R = CRE.rolling(D, MS.rich_cols(D, task), t)
    R.update(chronos_direct(task, dates))
    sign = 1.0 if task == "spike" else -1.0
    prob = lambda v: CM.calibrate(pd.DataFrame({"date": dates, "s": sign * v}), y)
    pool = {f"reg_{k}": prob(v) for k, v in R.items()}
    for k, (dd, p) in TA.candidates(task).items():
        pool[f"cls_{k}"] = pd.Series(p, index=np.asarray(dd)).reindex(dates).to_numpy(float)
    for nm, F in CM.features(task, dates).items():
        pool[f"cls_{nm}"] = CM.calibrate(F, y)
    ref = "reg_lgb_q50"
    devm = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y) & ~np.isnan(pool[ref])
    fill = lambda p: np.where(np.isnan(p), np.nanmedian(p[devm]), p)
    key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
    P = {k: fill(v) for k, v in pool.items() if np.isfinite(np.nanmedian(v[devm]))}
    regs = sorted([k for k in P if k.startswith("reg_")], key=lambda k: key(P[k]))
    allr = sorted(P, key=lambda k: key(P[k]))
    E = {"회귀 중앙값 전체": prob(np.nanmedian(np.vstack(list(R.values())), 0))}
    for k in range(2, 8):
        E[f"회귀 예측 평균 상위{k}"] = prob(np.nanmean([R[c[4:]] for c in regs[:k]], 0))
        E[f"확률 평균 상위{k}(전체)"] = np.mean([P[c] for c in allr[:k]], 0)
    E["회귀 스태킹 상위6"] = prob(CRE.stacking(D, R, t, [c[4:] for c in regs[:6]]))
    best_cls = [k for k in allr if k.startswith("cls_")][:2]
    E["회귀 중앙값 + 분류 상위2 평균"] = np.mean([fill(E["회귀 중앙값 전체"])] + [P[c] for c in best_cls], 0)
    for k, v in E.items():
        if np.isfinite(np.nanmedian(v[devm])):
            P[k] = fill(v)
    print(f"[{task}] 성분·앙상블 {len(P)}개 · {time.time() - t0:.0f}초")

    off = np.array([off_map.get(d) == "휴일" for d in dates])
    cd, clv = TA.current(task)
    cy = TA.labels(task, cd)
    cur = TA.day_stats(clv, cy, (cd >= DEV[0]) & (cd <= DEV[1]) & ~np.isnan(cy) & (clv != ""))
    slack = max(1, round(0.05 * cur["사건일"]))
    short = sorted(P, key=lambda k: key(P[k]))[:30]

    def opts(p, g):
        o = [(("고정",) + x, s) for x, s in MS.options_fixed(p, y, devm, g)]
        o += [(("매달",) + x, s) for x, s in MS.options_monthly(dates, p, y, devm, g)]
        return MS.pareto(o)

    def choose(names):
        best = None
        for n in names:
            p = P[n]
            oo, ow = opts(p, off), opts(p, ~off)
            for o1, s1 in oo:
                for o2, s2 in ow:
                    ha, hw = s1[0] + s2[0], s1[1] + s2[1]
                    if ha < cur["경보적중"] - slack or hw < cur["주의이상적중"] - slack:
                        continue
                    k_ = (s1[2] + s2[2] + s1[3] + s2[3], -hw, -ha, -ap_score(p[devm], y[devm]))
                    if best is None or k_ < best[0]:
                        best = (k_, n, o1, o2, (ha, hw, s1[2] + s2[2], s1[3] + s2[3]))
        return best

    def levels(n, o1, o2):
        p = P[n]
        lv = np.full(len(p), "정상", dtype=object)
        for g, o in ((off, o1), (~off, o2)):
            if o[0] == "매달":
                lg = MS.monthly_levels(dates, p, y, o[1], o[2], g)
                lv[g] = lg[g]
            else:
                ta, tw = o[1], o[2]
                lv[g & (p >= tw)] = "주의"
                lv[g & (p >= ta)] = "경보"
        lv[np.isnan(pool[ref])] = ""
        return lv

    rows = []
    for n in short:
        bb = choose([n])
        rows.append({"후보": n, "앙상블": n in E, "학습_FP@R80": key(P[n])[0], "학습_AP": round(-key(P[n])[1], 3),
                     "학습_헛경보(규칙)": None if bb is None else bb[0][0], "학습_적중(경보,주의이상)": None if bb is None else (bb[4][0], bb[4][1])})
    rows.sort(key=lambda r: (r["학습_헛경보(규칙)"] if r["학습_헛경보(규칙)"] is not None else 99999, r["학습_FP@R80"]))
    Dn = pd.DataFrame({"date": dates, "y": y})
    res = {"current_dev": cur, "slack": slack, "table_dev": rows,
           "test_current": W.seg_score(task, pd.DataFrame({"date": cd, "y": cy}), clv, *SETTINGS["test"], seg, off_map)}
    for tag, names in (("overall", short), ("ensemble", [n for n in short if n in E] or list(E))):
        bb = choose(names)
        if bb is None:
            res[tag] = None
            continue
        _, n, o1, o2, ds = bb
        res[tag] = {"후보": n, "휴일기준": o1, "평일기준": o2, "dev": dict(zip(["경보적중", "주의이상적중", "헛경보_경보", "헛경보_주의"], ds)),
                    "test": W.seg_score(task, Dn, levels(n, o1, o2), *SETTINGS["test"], seg, off_map)}
    print(f"[{task}] 학습 현재 {cur} (허용 {slack}) · {time.time() - t0:.0f}초")
    for r in rows[:14]:
        print(f"   {'[앙상블] ' if r['앙상블'] else ''}{r['후보']:<34} 헛경보(규칙) {r['학습_헛경보(규칙)']}  적중 {r['학습_적중(경보,주의이상)']}  FP@R80 {r['학습_FP@R80']}  AP {r['학습_AP']}")
    for tag in ("overall", "ensemble"):
        if res[tag]:
            s = res[tag]["test"]
            print(f"   {tag:<9} {res[tag]['후보']} · {res[tag]['휴일기준']} / {res[tag]['평일기준']} · 학습 {res[tag]['dev']}")
            print(f"             시험 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} / {s['구간']} · 헛경보 경보 {s['헛경보_경보일']} 주의 {s['헛경보_주의일']}")
    s = res["test_current"]
    print(f"   current   시험 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} / {s['구간']} · 헛경보 경보 {s['헛경보_경보일']} 주의 {s['헛경보_주의일']}")
    return res


def main():
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    report = {task: run(task, daily, seg, off_map) for task in ("spike", "jeju")}
    (OUT / "reg_ens_all_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
