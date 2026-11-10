# -*- coding: utf-8 -*-
"""
run_lear.py — LEAR (Lasso Estimated AutoRegressive) 벤치마크
==============================================================================
근거 : Lago, Marcjasz, De Schutter, Weron (2021) Applied Energy 293 — EPF 표준 벤치마크
  · 시간대(24)마다 선형 Lasso 1개, 규제 강도는 LassoLarsIC(AIC) 로 선택
  · 입력 : 대상일 i 기준 SMP(i−1, i−2, i−3, i−7 각 24시간) + 수요예측(i, i−1, i−7 각 24시간)
          + 요일 더미 + 휴일 여부(i, i−1)  → 177개
  · 변환 : 학습 구간의 중앙값·MAD 로 표준화 후 asinh (입력·출력 모두)
  · 재보정 : 매주 월요일, 학습 창 364일·1092일 두 가지 → 두 예측의 평균이 lear
산출
  outputs/ensemble/pred_lear.pkl, pred_lear364.pkl, pred_lear1092.pkl
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC

from common import OOS_START, OUT, load_hourly, to_matrix

warnings.filterwarnings("ignore")
WINDOWS = [364, 1092]


def design(P, Dm, off, dow):
    n = len(P)

    def sh(M, k):
        out = np.full_like(M, np.nan, dtype=float)
        out[k:] = M[:-k]
        return out

    dmy = np.eye(7)[dow]
    offf = off.astype(float)
    X = np.hstack([sh(P, 1), sh(P, 2), sh(P, 3), sh(P, 7), Dm, sh(Dm, 1), sh(Dm, 7), dmy,
                   offf[:, None], np.r_[0, offf[:-1]][:, None]])
    return X


def vst_fit(A):
    med = np.nanmedian(A, axis=0)
    mad = np.nanmedian(np.abs(A - med), axis=0) * 1.4826
    mad[~(mad > 1e-6)] = 1.0
    return med, mad


def main():
    h = load_hourly()
    dates = h["date"].iloc[::24].to_numpy()
    P, Dm = to_matrix(h, "smp"), to_matrix(h, "dem")
    off = h["off"].to_numpy().reshape(-1, 24)[:, 0]
    dow = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    X = design(P, Dm, off, dow)
    n_cont = X.shape[1] - 9                                # 요일·휴일 더미는 변환하지 않음
    last = int(np.where(~np.isnan(P).all(axis=1))[0].max())
    dt = pd.to_datetime(pd.Series(dates))
    cuts = [i for i in range(len(dates)) if dates[i] >= OOS_START and i <= last
            and (dt[i].dayofweek == 0 or dates[i] == OOS_START)] + [last + 1]
    preds = {w: np.full((len(dates), 24), np.nan) for w in WINDOWS}
    t0 = time.time()
    for a, b in zip(cuts[:-1], cuts[1:]):
        for w in WINDOWS:
            tr = np.arange(max(7, a - w), a)
            ok = ~np.isnan(X[tr]).any(1) & ~np.isnan(P[tr]).any(1)
            tr = tr[ok]
            xm, xs = vst_fit(X[tr, :n_cont])
            ym, ys = vst_fit(P[tr])
            tf = lambda A: np.hstack([np.arcsinh((A[:, :n_cont] - xm) / xs), A[:, n_cont:]])
            Xtr = tf(X[tr])
            Ytr = np.arcsinh((P[tr] - ym) / ys)
            te = np.arange(a, b)
            Xte = X[te].copy()
            bad = np.isnan(Xte)
            Xte[bad] = np.take(np.r_[xm, np.zeros(9)], np.where(bad)[1])
            Xte = tf(Xte)
            for j in range(24):
                m = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xtr, Ytr[:, j])   # LARS 경로의 AIC 최적 해
                preds[w][te, j] = np.sinh(m.predict(Xte)) * ys[j] + ym[j]
        if dates[a][:7] != dates[a - 7][:7] and dates[a][5:7] in ("01", "07"):
            print(f"  {dates[a]} {time.time() - t0:5.0f}초", flush=True)
    base = pd.DataFrame({"date": np.repeat(dates, 24), "hour": np.tile(np.arange(1, 25), len(dates)),
                         "y": P.ravel()})
    keep = base["date"] >= OOS_START
    for w in WINDOWS:
        base.assign(p=preds[w].ravel())[keep].to_pickle(OUT / f"pred_lear{w}.pkl")
    avg = np.nanmean(np.stack([preds[w] for w in WINDOWS]), axis=0)
    base.assign(p=avg.ravel())[keep].to_pickle(OUT / "pred_lear.pkl")
    print(f"완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
