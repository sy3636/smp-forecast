# -*- coding: utf-8 -*-
"""
run_lgbm.py — LightGBM 성분 모델 (매달 1일 그 이전 자료로만 재학습 → 그 달 D+1 예측)
==============================================================================
변형
  | 이름       | 목적함수 | 타깃 변환               | 비고 |
  |------------|---------|-------------------------|------|
  | lgb_l1     | L1(MAE) | VST asinh               | 조건부 중앙값 — MAE 정렬 |
  | lgb_l2     | L2      | VST asinh               | 조건부 평균 |
  | lgb_diff   | L2      | y − 전일 동시간          | 수준 변화만 학습 |
  | lgb_stack  | L1      | VST asinh               | + Chronos-2 분위수(표본 외)를 변수로 투입 |
  · VST 계수(중앙값·MAD)는 학습 구간에서만 산출 (scripts/step4_train_ensemble.py 와 같은 식)

실행
  python experiments/ensemble/run_lgbm.py                       # 기본 3종
  python experiments/ensemble/run_lgbm.py --variants lgb_stack --chronos dem_off
  python experiments/ensemble/run_lgbm.py --refit weekly        # 매주 재학습
산출
  outputs/ensemble/pred_<변형>[_weekly].pkl — date, hour, y, p(예측)
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd

from common import OOS_START, OUT, load_lightgbm
from features import build, feature_cols

lgb = load_lightgbm()

VARIANTS = {
    "lgb_l1": dict(obj="l1", tf="vst"),
    "lgb_l2": dict(obj="l2", tf="vst"),
    "lgb_diff": dict(obj="l2", tf="diff"),
    "lgb_stack": dict(obj="l1", tf="vst", stack="dem_off"),
    # 미세조정 Chronos(2023년~ 산출) 분위수를 변수로 — 학습 표본 확보 위해 2023-07 부터 예측
    "lgb_stack_ft": dict(obj="l1", tf="vst", stack="ft_lora", start="2023-07-01"),
}
PARAMS = dict(learning_rate=0.05, n_estimators=800, num_leaves=63, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              verbose=-1, n_jobs=8, random_state=42)


def chronos_features(X, name):
    with open(OUT / f"chronos_{name}.pkl", "rb") as f:
        c = pickle.load(f)
    lv = c["q_levels"]
    q = c["q"]
    day_idx = {d: i for i, d in enumerate(c["dates"])}
    di = X["date"].map(day_idx)
    miss = di.isna().to_numpy()                        # Chronos 산출 범위 밖 날짜 → NaN
    di = di.fillna(0).astype(int).to_numpy()
    hi = X["hour"].to_numpy(int) - 1
    out = pd.DataFrame(index=X.index)
    for lvl in [0.1, 0.5, 0.9]:
        out[f"c_q{int(lvl * 100)}"] = q[di, hi, lv.index(lvl)]
    q50 = q[:, :, lv.index(0.5)]
    out["c_day_mean"] = np.nanmean(q50, 1)[di]
    out["c_mid_min"] = np.nanmin(q50[:, 10:15], 1)[di]
    out["c_spread"] = out["c_q90"] - out["c_q10"]
    out.loc[miss] = np.nan
    return out


def cut_dates(dates, refit):
    days = pd.to_datetime(pd.Series(sorted(set(dates))))
    days = days[days >= OOS_START]
    if refit == "weekly":
        cuts = days[days.dt.dayofweek == 0]           # 매주 월요일
        cuts = pd.concat([days.iloc[:1], cuts]).drop_duplicates()
    else:
        cuts = days[days.dt.day == 1]
    return list(cuts.dt.strftime("%Y-%m-%d")) + ["9999-12-31"]


def fit_predict(Xtr, ytr, p1tr, Xte, p1te, spec):
    if spec["tf"] == "vst":
        med = float(np.median(ytr))
        mad = float(np.median(np.abs(ytr - med))) * 1.4826 or 1.0
        z = np.arcsinh((ytr - med) / mad)
        m = lgb.LGBMRegressor(objective=spec["obj"], **PARAMS).fit(Xtr, z)
        return np.sinh(m.predict(Xte)) * mad + med
    ok = ~np.isnan(p1tr)
    m = lgb.LGBMRegressor(objective=spec["obj"], **PARAMS).fit(Xtr[ok], (ytr - p1tr)[ok])
    return p1te + m.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["lgb_l1", "lgb_l2", "lgb_diff"])
    ap.add_argument("--refit", default="monthly", choices=["monthly", "weekly"])
    ap.add_argument("--from", dest="since", default=None,
                    help="이 날짜 이후 재학습 구간만 다시 계산하고 그 전 예측은 기존 파일 값을 유지 (자료 추가 시)")
    # 2026-11-09 팀 작업본 추가 — 2020년 자료로 학습 (기본값은 팀 원래 설정과 같음)
    ap.add_argument("--train-from", default="2021-01-08", help="학습 자료 시작일 (2020 자료를 쓰려면 2020-01-08)")
    ap.add_argument("--tag", default="", help="산출 파일 이름 뒤에 붙일 꼬리표 — 팀 원래 결과를 덮어쓰지 않게")
    ap.add_argument("--stack-allow-missing", action="store_true",
                    help="적층형: Chronos 분위수가 없는 날(2021년 이전)도 학습에 씀 (LightGBM 이 빈 값을 처리)")
    args = ap.parse_args()

    X = build()
    X = X[X["date"] >= args.train_from].reset_index(drop=True)
    cols = feature_cols(X)
    cfs = {VARIANTS[v]["stack"]: chronos_features(X, VARIANTS[v]["stack"]) for v in args.variants if VARIANTS[v].get("stack")}
    cuts = cut_dates(X["date"], args.refit)
    suffix = "_weekly" if args.refit == "weekly" else ""
    preds = {v: np.full(len(X), np.nan) for v in args.variants}
    if args.since:                                     # 기존 예측 재사용 — 그 이전 구간은 입력이 바뀌지 않음
        for v in args.variants:
            old = pd.read_pickle(OUT / f"pred_{v}{suffix}.pkl")[["date", "hour", "p"]]
            keep = X[["date", "hour"]].merge(old, on=["date", "hour"], how="left")["p"].to_numpy()
            preds[v] = np.where(X["date"].to_numpy() < args.since, keep, np.nan)
        start_i = max(i for i, c in enumerate(cuts[:-1]) if c <= args.since)
        cuts = cuts[start_i:]
    t0 = time.time()
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr = (X["date"] < a) & X["y"].notna()
        te = (X["date"] >= max(a, args.since or a)) & (X["date"] < b)
        if not te.any():
            continue
        for v in args.variants:
            spec = VARIANTS[v]
            if a < spec.get("start", OOS_START):
                continue
            if spec.get("stack"):
                cf = cfs[spec["stack"]]
                Xv = pd.concat([X[cols], cf], axis=1)
                tr_v = tr if args.stack_allow_missing else tr & cf["c_q50"].notna()
            else:
                Xv, tr_v = X[cols], tr
            preds[v][te.to_numpy()] = fit_predict(
                Xv[tr_v].to_numpy(float), X.loc[tr_v, "y"].to_numpy(), X.loc[tr_v, "p1"].to_numpy(),
                Xv[te].to_numpy(float), X.loc[te, "p1"].to_numpy(), spec)
        if a[5:7] in ("01", "07") and a[8:10] <= "07":
            print(f"  {a} 까지 {time.time() - t0:5.0f}초", flush=True)
    for v in args.variants:
        df = X[["date", "hour", "y"]].assign(p=preds[v])
        df = df[df["date"] >= OOS_START]
        df.to_pickle(OUT / f"pred_{v}{suffix}{args.tag}.pkl")
    print(f"완료 {time.time() - t0:.0f}초 — {', '.join(args.variants)}{suffix}{args.tag}")


if __name__ == "__main__":
    main()
