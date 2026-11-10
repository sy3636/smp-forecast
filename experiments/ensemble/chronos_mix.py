# -*- coding: utf-8 -*-
"""
chronos_mix.py — Chronos-2 예측을 경보 성분으로 넣어 앙상블 시험 (2026-11-08, 팀 작업본)
==============================================================================
Chronos-2 는 다시 돌리지 않음(run_chronos*.py 실행 금지). 팀이 저장한 분위수 예측만 씀.
  · 육지 급락·제주 급락 (전날 15시) : chronos_off      — D+1 수요예측 없이
  · 육지 급등 (전날 21시)            : chronos_dem_off · chronos_ft_lora — 수요예측 사용
Chronos 성분 : 분위수 11개로 시간별 확률(가격 < 30원, 또는 ≥ 급등 기준)을 구해 하루 변수(최대·상위3 평균·기대 시간 수)로 만들고,
              매달 1일 그 이전 자료로만 로지스틱으로 보정 → 그 달 확률 (표본 외)
앙상블 : 기존 후보(warn_search · crash_search · 현재 모델) + Chronos 성분 섞기(0.1~0.5), 상위 k 평균
고르기 (학습 구간 2022–2024 만)
  ① FP@R80 최소 (이전 탐색과 같은 규칙)
  ② tune_alerts 규칙: 적중 감소 ≤ max(1, 사건일 5%) 안에서 헛경보(경보+주의) 합계 최소, 기준 재현율 함께 고름
시험 구간은 고른 것만 계산.  실행 : python experiments/ensemble/chronos_mix.py
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import events as EVM
import tune_alerts as TA
import warn_search as W
from common import OUT, SETTINGS, rd
from warnings_model import EVAL_FROM, ap_score, fp_at_recall, recency

warnings.filterwarnings("ignore")
DEV = TA.DEV


def chronos(name, dates):
    c = pickle.load(open(OUT / f"chronos_{name}.pkl", "rb"))
    idx = {d: i for i, d in enumerate(c["dates"])}
    q = np.full((len(dates), 24, len(c["q_levels"])), np.nan)
    for i, d in enumerate(dates):
        if d in idx:
            q[i] = c["q"][idx[d]]
    return q, np.array(c["q_levels"])


def cdf(q, lv, v):
    """분위수 → P(가격 < v). q:(n,24,k), v:(n,24) 또는 스칼라. 범위 밖은 끝 분위의 절반으로"""
    v = np.broadcast_to(v, q.shape[:2])
    out = np.full(q.shape[:2], np.nan)
    for i in range(q.shape[0]):
        for h in range(24):
            qq = q[i, h]
            if np.isnan(qq).any() or np.isnan(v[i, h]):
                continue
            qs = np.maximum.accumulate(qq)
            out[i, h] = np.interp(v[i, h], qs, lv, left=lv[0] / 2, right=1 - (1 - lv[-1]) / 2)
    return out


def features(task, dates):
    F = pd.DataFrame({"date": dates})
    if task in ("crash", "jeju"):
        q, lv = chronos("off", dates)
        p30 = cdf(q, lv, 30.0)
        mid = p30[:, 10:15]
        F["c_pmax"] = np.nanmax(mid, 1)
        F["c_ptop"] = np.sort(np.nan_to_num(mid, nan=0), 1)[:, -3:].mean(1)
        F["c_pexp"] = np.nansum(mid, 1)
        if task == "jeju":
            p0 = cdf(q, lv, 0.5)
            F["c_p0max"] = np.nanmax(p0, 1)
            F["c_pall"] = np.nanmax(p30, 1)
        return {"chr_off": F}
    S = EVM.load_settings()
    A, _, _ = EVM.load_actuals(S)
    ad = A["date"].iloc[::24].to_numpy()
    thr = EVM.baseline(A["land"].to_numpy(float).reshape(-1, 24), S["median_days"]) + S["spike_add_won"]
    ti = {d: i for i, d in enumerate(ad)}
    T = np.array([thr[ti[d]] if d in ti else np.full(24, np.nan) for d in dates])
    out = {}
    for nm in ("dem_off", "ft_lora"):
        q, lv = chronos(nm, dates)
        ps = 1 - cdf(q, lv, T)                                   # P(가격 ≥ 급등 기준)
        G = pd.DataFrame({"date": dates})
        G["c_pmax"] = np.nanmax(ps, 1)
        G["c_ptop"] = np.sort(np.nan_to_num(ps, nan=0), 1)[:, -3:].mean(1)
        G["c_pexp"] = np.nansum(ps, 1)
        out[f"chr_{nm}"] = G
    return out


def calibrate(F, y):
    """매달 1일 그 이전 자료로 로지스틱 보정 → 표본 외 확률"""
    cols = [c for c in F.columns if c != "date"]
    X = F[cols].to_numpy(float)
    ok = ~np.isnan(X).any(1) & ~np.isnan(y)
    dnum = pd.to_datetime(F["date"]).to_numpy().astype("datetime64[D]").astype(int)
    p = np.full(len(F), np.nan)
    for ym in sorted(F.loc[F["date"] >= EVAL_FROM, "date"].str[:7].unique()):
        start = int(np.datetime64(ym + "-01", "D").astype(int))
        tr = ok & (dnum < start)
        te = (F["date"].str[:7] == ym).to_numpy() & ~np.isnan(X).any(1)
        if y[tr].sum() < 1 or not te.any():
            continue
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        m.fit(X[tr], y[tr].astype(int), sample_weight=recency(start - dnum[tr]))
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main():
    seg, daily = rd(OUT / "events_segments.csv"), rd(OUT / "events_daily.csv")
    off_map = dict(zip(daily["date"], daily["구분"]))
    a, b = SETTINGS["test"]
    report = {}
    for task in ("crash", "spike", "jeju"):
        monthly = task == "jeju"
        base = TA.candidates(task)
        dates = np.asarray(next(iter(base.values()))[0])
        al = lambda dd, p: pd.Series(p, index=dd).reindex(dates).to_numpy(float)
        pool = {k: al(np.asarray(d), p) for k, (d, p) in base.items()}
        y = TA.labels(task, dates)
        for nm, F in features(task, dates).items():
            pool[nm] = calibrate(F, y)
        devm = (dates >= DEV[0]) & (dates <= DEV[1]) & ~np.isnan(y) & ~np.isnan(pool[next(iter(base))])
        fill = lambda p: np.where(np.isnan(p), np.nanmedian(p[devm]), p)
        P = {k: fill(v) for k, v in pool.items()}
        key = lambda p: (fp_at_recall(p[devm], y[devm]), -ap_score(p[devm], y[devm]))
        chr_names = [k for k in P if k.startswith("chr_")]
        ranked = sorted([k for k in base], key=lambda k: key(P[k]))[:5]
        C = {k: P[k] for k in list(base) + chr_names}
        for c in ranked:                                          # 기존 상위 5개 × Chronos 섞기
            for cn in chr_names:
                for w in (0.1, 0.2, 0.3, 0.4, 0.5):
                    C[f"{c} + {cn} {w:.1f}"] = (1 - w) * P[c] + w * P[cn]
        pool_rank = sorted(list(base) + chr_names, key=lambda k: key(P[k]))
        for k in range(2, 6):
            C[f"평균 상위{k}(Chronos 포함 후보)"] = np.mean([P[c] for c in pool_rank[:k]], 0)
        rows = sorted([{"후보": n, "학습_FP@R80": key(p)[0], "학습_AP": round(-key(p)[1], 3), "Chronos": ("chr_" in n)}
                       for n, p in C.items()], key=lambda r: (r["학습_FP@R80"], -r["학습_AP"]))
        # ② 기준 재현율 함께 — tune_alerts 규칙
        cd, clv = TA.current(task)
        cy = TA.labels(task, cd)
        cdev = (cd >= DEV[0]) & (cd <= DEV[1]) & ~np.isnan(cy) & (clv != "")
        cur_dev = TA.day_stats(clv, cy, cdev)
        slack = max(1, round(0.05 * cur_dev["사건일"]))
        short = [r["후보"] for r in rows[:25]]
        best = None
        for n in short:
            p = C[n]
            for ra in TA.RA:
                for rw in [r for r in TA.RW if r >= ra]:
                    lv = TA.levels(dates, p, y, ra, rw, monthly, devm)
                    s = TA.day_stats(lv, y, devm & (lv != ""))
                    if s["경보적중"] < cur_dev["경보적중"] - slack or s["주의이상적중"] < cur_dev["주의이상적중"] - slack:
                        continue
                    k_ = (s["헛경보_경보"] + s["헛경보_주의"], -s["주의이상적중"], -s["경보적중"])
                    if best is None or k_ < best[0]:
                        best = (k_, n, ra, rw, s, lv)
        _, bn, ra, rw, bs, blv = best
        Dn = pd.DataFrame({"date": dates, "y": y})
        Dc = pd.DataFrame({"date": cd, "y": cy})
        res = {"top_dev": rows[:12], "chronos_alone": [r for r in rows if r["후보"] in chr_names],
               "tuned": {"후보": bn, "r_alarm": ra, "r_watch": rw, "dev": bs, "dev_current": cur_dev, "slack": slack},
               "test_current_seg": W.seg_score(task, Dc, clv, a, b, seg, off_map),
               "test_tuned_seg": W.seg_score(task, Dn, blv, a, b, seg, off_map)}
        report[task] = res
        print(f"\n=== {task}")
        print("  Chronos 단독:", [(r["후보"], r["학습_FP@R80"], r["학습_AP"]) for r in res["chronos_alone"]])
        print("  학습 FP@R80 상위:")
        for r in rows[:8]:
            print(f"    {r['후보']:<52} FP@R80 {r['학습_FP@R80']:>4}  AP {r['학습_AP']}")
        print(f"  기준 함께 고른 조합: {bn} · 경보 {ra} · 주의 {rw} · 학습 {bs} (현재 {cur_dev}, 허용 {slack})")
        for nm in ("test_current_seg", "test_tuned_seg"):
            s = res[nm]
            print(f"  {nm:<16} 구간 {s['구간']} · 경보 {s['성공']} 주의 {s['부분']} 놓침 {s['실패'] + s['예측 없음']} · 헛경보 경보 {s['헛경보_경보일']}일 주의 {s['헛경보_주의일']}일")
    (OUT / "chronos_mix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
