# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 하이퍼파라미터 탐색
================================================
목적 : 시계열 교차검증 기준으로 모델 하이퍼파라미터를 탐색하여 최적 조합 확정
방식 : 무작위 탐색(random search) — 격자 탐색 대비 동일 시행횟수에서 탐색 효율 우수
입력 : data/processed/smp_features_hourly.csv, feature_dictionary.csv
출력 : config/model_params_{target}.json   (최적 파라미터 — step4_train에서 자동 적용)
       logs/step4_tune_{일시}.csv          (전체 시행 이력)

설계
  · 검증 : 확장창 시계열 교차검증(기본 3분할, 최근 구간 우선)
  · 지표 : MAPE(주지표) 최소화, 동률 시 RMSE
  · 누설 : 홀드아웃(2026년) 구간은 탐색에 일절 미사용

버전 : v1.0 (2026-08-01)
실행 : python scripts\\step4_tune.py                    (기본 20회 시행)
       python scripts\\step4_tune.py --n-iter 40 --folds 3
       python scripts\\step4_tune.py --target smp_jeju
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import MAPE_FLOOR, set_mape_floor, feat_paths, metrics, TRAIN_START, CV_END  # noqa: E402
from step4_train import EXCLUDE, CATEGORICAL, load_model_class  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# 탐색 공간
SPACE_LGB = {
    "n_estimators": [400, 700, 1000, 1500, 2000],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
    "num_leaves": [31, 63, 95, 127, 191],
    "min_child_samples": [10, 20, 30, 50, 80],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_lambda": [0.0, 0.5, 1.0, 5.0, 10.0],
}
SPACE_SK = {
    "max_iter": [400, 700, 1000, 1500],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
    "max_leaf_nodes": [31, 63, 95, 127],
    "min_samples_leaf": [10, 20, 30, 50, 80],
    "l2_regularization": [0.0, 0.5, 1.0, 5.0, 10.0],
    "max_features": [0.6, 0.8, 1.0],
}


def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def sample(space, rng):
    return {k: rng.choice(v) for k, v in space.items()}


def build(kind, lgb, params, seed):
    if kind == "lightgbm":
        return lgb.LGBMRegressor(objective="regression", random_state=seed,
                                 n_jobs=-1, verbose=-1, subsample_freq=1, **params)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=seed, **params)


def main():
    ap = argparse.ArgumentParser(description="step4 하이퍼파라미터 탐색")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--n-iter", type=int, default=20, help="시행 횟수(기본 20)")
    ap.add_argument("--folds", type=int, default=3, help="교차검증 분할수(기본 3)")
    ap.add_argument("--min-train-days", type=int, default=730)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR,
                    help="MAPE 분모 하한(원/kWh). 기본 20")
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    SRC, DICT = feat_paths(t)
    for p in (SRC, DICT):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    kind, lgb = load_model_class()
    space = SPACE_LGB if kind == "lightgbm" else SPACE_SK
    rng = random.Random(args.seed)

    print("=" * 78)
    print(" step4 하이퍼파라미터 탐색 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 타깃 : {t} / 엔진 : {kind} / 시행 {args.n_iter}회 × {args.folds}분할")
    print("=" * 78)

    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    cats = [c for c in CATEGORICAL if c in feats]
    for c in cats:
        df[c] = df[c].astype("category")
    feats = [c for c in feats if c in cats or is_num(df[c])]

    cv = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].dropna(subset=[t]).copy()
    days = sorted(cv["date"].unique())
    n_min = min(args.min_train_days, len(days) - args.folds * 30)
    size = max(30, (len(days) - n_min) // args.folds)
    folds = []
    for k in range(args.folds):
        tr_d = set(days[: n_min + size * k])
        va_d = set(days[n_min + size * k: n_min + size * (k + 1)])
        if va_d:
            folds.append((cv[cv["date"].isin(tr_d)], cv[cv["date"].isin(va_d)]))

    print(f"\n■ 탐색 조건")
    print(f"  · 투입 변수 : {len(feats)}개")
    print(f"  · 학습 구간 : {cv['date'].min()} ~ {cv['date'].max()} ({len(cv):,}행)")
    print(f"  · 분할 수   : {len(folds)}개 (검증창 {size}일)")
    print(f"  · 탐색 공간 : {len(space)}개 파라미터")

    # 기준(현행 기본값) 성능
    print(f"\n■ 시행 결과")
    print(f"  {'시행':<5}{'MAPE':>7}{'RMSE':>8}{'소요':>7}  파라미터")
    print("  " + "-" * 74)

    trials, best = [], None
    for i in range(args.n_iter):
        p = sample(space, rng)
        t0 = time.time()
        ms = []
        for tr, va in folds:
            m = build(kind, lgb, p, args.seed)
            m.fit(tr[feats], tr[t])
            ms.append(metrics(va[t], pd.Series(m.predict(va[feats]), index=va.index)))
        mape = float(np.nanmean([x["MAPE"] for x in ms]))
        rmse = float(np.nanmean([x["RMSE"] for x in ms]))
        el = time.time() - t0
        rec = {"시행": i + 1, "MAPE": round(mape, 4), "RMSE": round(rmse, 3),
               "소요(초)": round(el, 1), **{k: str(v) for k, v in p.items()}}
        trials.append(rec)
        mark = ""
        if best is None or mape < best["MAPE"]:
            best = {"MAPE": mape, "RMSE": rmse, "params": p}
            mark = "  ← 갱신"
        brief = ", ".join(f"{k[:12]}={v}" for k, v in list(p.items())[:3])
        print(f"  {i + 1:<5}{mape:>7.3f}{rmse:>8.2f}{el:>7.1f}  {brief}{mark}")

    # ── 결과 ───────────────────────────────────
    print("\n■ 최적 파라미터")
    for k, v in best["params"].items():
        print(f"  · {k:<20}{v}")
    print(f"\n  · 교차검증 MAPE {best['MAPE']:.3f}% / RMSE {best['RMSE']:.2f}")

    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    pd.DataFrame(trials).sort_values("MAPE").to_csv(
        os.path.join(LOG_DIR, f"step4_tune_{t}_{ts}.csv"),
        index=False, encoding="utf-8-sig")

    cfg = os.path.join(CONFIG_DIR, f"model_params_{t}.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"engine": kind, "target": t, "tuned": ts,
                   "cv_mape": round(best["MAPE"], 4),
                   "params": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                              for k, v in best["params"].items()}},
                  f, ensure_ascii=False, indent=2)

    print(f"\n  → 파라미터 저장 {cfg}")
    print(f"  → 시행 이력 {os.path.join(LOG_DIR, f'step4_tune_{t}_{ts}.csv')}")
    print("=" * 78)
    print(" ※ step4_train.py 재실행 시 본 파라미터가 자동 적용됨")
    print("=" * 78)


if __name__ == "__main__":
    main()
