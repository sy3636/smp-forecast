# -*- coding: utf-8 -*-
"""
tune_alerts.py — 헛경보 줄이기: 모델 후보 × 주의·경보 기준(재현율 목표) 함께 고르기 (2026-11-08, 팀 작업본)
==============================================================================
고르는 규칙 (학습 구간 2022–2024 의 날짜 단위 성적만 사용)
  · 적중이 현재보다 많이 줄면 안 됨: 학습 구간에서 「경보 적중일」과 「주의 이상 적중일」이 각각
    현재 − max(1, 사건일의 5%) 이상
  · 그 안에서 헛경보 날(경보 + 주의) 합계가 가장 적은 조합 → 동률이면 적중 많은 쪽
  · 기준 = 학습 구간 사례의 r_alarm(경보)·r_watch(주의)를 잡는 확률. r_alarm 0.30~0.90, r_watch r_alarm~0.95 (0.05 간격)
  · 제주는 기준값을 매달 그 전까지의 사례로 다시 정함(현재 방식과 같음)
후보 : 육지 급락 = crash_search_probs.pkl (현재 2개 모델 조합은 기준선으로만), 육지 급등 = 현재(warnings_spike) + warn_search 후보,
       제주 = 현재(warnings_jeju) + warn_search 후보. 「순위 평균」은 미래 자료가 섞여 제외.
시험 구간(2025-01-01~2026-10-31) 성적은 고른 조합과 현재에 대해서만 계산 — 고르는 데 쓰지 않음.
실행 : python experiments/ensemble/tune_alerts.py      산출 : outputs/ensemble/tune_alerts_report.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import events as EVM
import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import thr_at_recall

warnings.filterwarnings("ignore")
DEV = ("2022-01-01", "2024-12-31")
RA = [round(x, 2) for x in np.arange(0.30, 0.901, 0.05)]
RW = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]


def labels(task, dates):
    d = rd(OUT / "events_daily.csv").set_index("date")
    col = {"crash": "급락구간", "spike": "급등구간", "jeju": "제주0원구간"}[task]
    ok = pd.Series(dates).map(d["판정"]).fillna(False).astype(bool).to_numpy()
    n = pd.Series(dates).map(d[col]).to_numpy(float)
    return np.where(ok, (n > 0).astype(float), np.nan)


def levels(dates, p, y, ra, rw, monthly, devm):
    dates = pd.Series(dates)
    if not monthly:
        ta, tw = thr_at_recall(p[devm], y[devm], ra), thr_at_recall(p[devm], y[devm], rw)
        return np.where(np.isnan(p), "", np.where(p >= ta, "경보", np.where(p >= tw, "주의", "정상")))
    TA, TW = np.full(len(p), np.nan), np.full(len(p), np.nan)
    base = ~np.isnan(p) & ~np.isnan(y)
    for ym in sorted(dates[~np.isnan(p)].str[:7].unique()):
        past = base & (dates < ym + "-01").to_numpy()
        if y[past].sum() < 1:
            continue
        m = (dates.str[:7] == ym).to_numpy()
        TA[m], TW[m] = thr_at_recall(p[past], y[past], ra), thr_at_recall(p[past], y[past], rw)
    return np.where(np.isnan(TA) | np.isnan(p), "", np.where(p >= TA, "경보", np.where(p >= TW, "주의", "정상")))


def day_stats(lv, y, m):
    yy, l = y[m], lv[m]
    a, wa = l == "경보", np.isin(l, ["경보", "주의"])
    return {"사건일": int(yy.sum()), "경보적중": int((a & (yy == 1)).sum()), "주의이상적중": int((wa & (yy == 1)).sum()),
            "헛경보_경보": int((a & (yy == 0)).sum()), "헛경보_주의": int(((l == "주의") & (yy == 0)).sum())}


def candidates(task):
    """{이름: (dates, p)} — 날짜 정렬 확률"""
    C = {}
    if task == "crash":
        pr = pd.read_pickle(OUT / "crash_search_probs.pkl")
        for c in pr.columns:
            if c not in ("date", "y"):
                C[c] = (pr["date"].to_numpy(), pr[c].to_numpy(float))
    else:
        cur = rd(OUT / ("warnings_spike.csv" if task == "spike" else "warnings_jeju.csv"))
        C["현재 모델"] = (cur["date"].to_numpy(), cur["확률"].to_numpy(float))
        pr = pd.read_pickle(OUT / "warn_search_probs.pkl")[task]
        for c in pr.columns:
            if c not in ("date", "y") and not c.startswith("순위"):
                C[c] = (pr["date"].to_numpy(), pr[c].to_numpy(float))
    return C


def current(task):
    """현재 대시보드 단계 (날짜, 단계) — 육지 급락은 휴일 모델 + 평일 모델"""
    S = EVM.load_settings()
    days = pd.date_range("2022-01-01", S["data_end"]).strftime("%Y-%m-%d").to_numpy()
    return days, EVM.load_levels(S, days)[task]


def main():
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    a, b = SETTINGS["test"]
    report = {}
    for task in ("crash", "spike", "jeju"):
        monthly = task == "jeju"
        cd, clv = current(task)
        cy = labels(task, cd)
        cdev = (cd >= DEV[0]) & (cd <= DEV[1]) & ~np.isnan(cy) & (clv != "")
        ctest = (cd >= a) & (cd <= b) & ~np.isnan(cy) & (clv != "")
        cur_dev = day_stats(clv, cy, cdev)
        slack = max(1, round(0.05 * cur_dev["사건일"]))
        rows = []
        for name, (dates, p) in candidates(task).items():
            y = labels(task, dates)
            dates = np.asarray(dates)
            devm = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y) & ~np.isnan(p)
            if y[devm].sum() < 1:
                continue
            for ra in RA:
                for rw in [r for r in RW if r >= ra]:
                    lv = levels(dates, p, y, ra, rw, monthly, devm)
                    s = day_stats(lv, y, devm & (lv != ""))
                    ok = s["경보적중"] >= cur_dev["경보적중"] - slack and s["주의이상적중"] >= cur_dev["주의이상적중"] - slack
                    rows.append({"후보": name, "r_alarm": ra, "r_watch": rw, **s, "조건충족": ok,
                                 "헛경보합": s["헛경보_경보"] + s["헛경보_주의"]})
        R = pd.DataFrame(rows)
        okR = R[R["조건충족"]].sort_values(["헛경보합", "주의이상적중", "경보적중"], ascending=[True, False, False])
        best = okR.iloc[0]
        dates, p = candidates(task)[best["후보"]]
        dates = np.asarray(dates)
        y = labels(task, dates)
        devm = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y) & ~np.isnan(p)
        lv = levels(dates, p, y, best["r_alarm"], best["r_watch"], monthly, devm)
        testm = (dates >= a) & (dates <= b) & ~np.isnan(y) & (lv != "")
        Dn = pd.DataFrame({"date": dates, "y": y})
        Dc = pd.DataFrame({"date": cd, "y": cy})
        res = {"slack": slack, "dev_current": cur_dev, "dev_chosen": {k: (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best.items()},
               "top_dev": okR.head(8).to_dict("records"),
               "test_current_days": day_stats(clv, cy, ctest), "test_chosen_days": day_stats(lv, y, testm),
               "test_current_seg": W.seg_score(task, Dc, clv, a, b, seg, off_map),
               "test_chosen_seg": W.seg_score(task, Dn, lv, a, b, seg, off_map)}
        report[task] = res
        print(f"\n=== {task} · 허용 적중 감소 {slack}일 · 학습 현재 {cur_dev}")
        print(okR.head(6)[["후보", "r_alarm", "r_watch", "경보적중", "주의이상적중", "헛경보_경보", "헛경보_주의", "헛경보합"]].to_string(index=False))
        for nm in ("test_current_seg", "test_chosen_seg"):
            s = res[nm]
            print(f"  {nm:<17} 구간 {s['구간']} · 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} · 헛경보 경보 {s['헛경보_경보일']}일 주의 {s['헛경보_주의일']}일")
    (OUT / "tune_alerts_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
