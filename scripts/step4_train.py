# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 학습 모델(시간별 SMP 예측)
================================================
목적 : 운영 가용 변수만으로 시간별 SMP 예측 모델을 학습하고,
       벤치마크(step4_baseline) 대비 개선율을 산출
입력 : data/processed/smp_features_hourly.csv
       data/processed/feature_dictionary.csv   (운영가용 = Y 변수만 사용)
       data/processed/baseline_predictions.csv (개선율 비교용, 선택)
출력 : ml/models/smp_{target}_lgbm.pkl          (최종 모델)
       data/processed/model_predictions_{target}.csv  (예측값)
       logs/step4_train_summary_{일시}.csv      (성능)
       logs/step4_train_importance_{일시}.csv   (변수 중요도)

설계
  · 누설 방지 : 변수사전의 `운영가용 = Y` 변수만 투입(실적계 변수 전면 배제)
  · 검증      : 확장창(expanding window) 시계열 교차검증 5분할
  · 홀드아웃  : 2026-01-01 이후 1회 평가
  · 모델      : LightGBM(미설치 시 scikit-learn HistGradientBoosting으로 자동 대체)

버전 : v1.0 (2026-07-31)
실행 : python scripts\\step4_train.py
       python scripts\\step4_train.py --target smp_jeju
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import MAPE_FLOOR, mape_floor, set_mape_floor, feat_paths, metrics, TRAIN_START, CV_END, HOLD_START  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")


def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def base_pred_path(t):
    """벤치마크 예측 파일(타깃별 분리) — 타깃 혼용 비교 방지."""
    return os.path.join(PROCESSED, f"baseline_predictions_{t}.csv")

# 투입 제외 변수(식별자·외삽 위험·타깃 계열)
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju",
           "year",              # 홀드아웃에 미학습 연도가 등장 → 외삽 위험
           "unit_id", "unit_name", "grp", "pv_source", "holiday_name"}
CATEGORICAL = ["daytype", "season"]


def load_model_class():
    """LightGBM 우선, 미설치 시 scikit-learn 대체."""
    try:
        import lightgbm as lgb
        return "lightgbm", lgb
    except ImportError:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
            return "sklearn", None
        except ImportError:
            sys.exit("[중단] lightgbm·scikit-learn 미설치 → pip install -r requirements.txt")


DEFAULT_LGB = dict(n_estimators=1200, learning_rate=0.05, num_leaves=63,
                   min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
                   reg_lambda=1.0)
DEFAULT_SK = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
                  min_samples_leaf=30, l2_regularization=1.0)


def load_tuned_params(kind, target):
    """step4_tune 산출 파라미터가 있으면 적용, 없으면 기본값."""
    import json
    p = os.path.join(BASE_DIR, "config", f"model_params_{target}.json")
    if os.path.exists(p):
        try:
            c = json.load(open(p, encoding="utf-8"))
            if c.get("engine") == kind:
                return c["params"], f"탐색값 적용({c.get('tuned', '')})"
        except Exception:
            pass
    return (DEFAULT_LGB if kind == "lightgbm" else DEFAULT_SK), "기본값"


def make_model(kind, lgb, seed=42, params=None):
    if kind == "lightgbm":
        p = dict(DEFAULT_LGB if params is None else params)
        for k in ("n_estimators", "num_leaves", "min_child_samples"):
            if k in p:
                p[k] = int(p[k])
        obj = p.pop("objective", "regression")   # 목적함수 지정 시 우선 적용
        return lgb.LGBMRegressor(objective=obj, subsample_freq=1,
                                 random_state=seed, n_jobs=-1, verbose=-1, **p)
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = dict(DEFAULT_SK if params is None else params)
    for k in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
        if k in p:
            p[k] = int(p[k])
    return HistGradientBoostingRegressor(random_state=seed, **p)


def main():
    ap = argparse.ArgumentParser(description="step4 학습 모델")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train-days", type=int, default=730,
                    help="최초 분할의 최소 학습기간(일). 기본 730일(2년)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR,
                    help="MAPE 분모 하한(원/kWh). 기본 20")
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    SRC, DICT = feat_paths(t)
    for p in (SRC, DICT):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}\n       step3_features.py 선행 실행 필요")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    kind, lgb = load_model_class()
    tuned, src_p = load_tuned_params(kind, t)

    print("=" * 78)
    print(" step4 학습 모델 - 시간별 SMP 예측 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t} / 엔진 : {kind}")
    print(f" 파라미터 : {src_p}")
    print("=" * 78)

    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")

    # ── ① 변수 선정 ────────────────────────────
    print("\n■ ① 변수 선정 (운영가용 = Y)")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    cats = [c for c in CATEGORICAL if c in feats]
    for c in cats:
        df[c] = df[c].astype("category")
    num = [c for c in feats if c not in cats]
    drop_obj = [c for c in num if not is_num(df[c])]
    if drop_obj:
        feats = [c for c in feats if c not in drop_obj]
        num = [c for c in num if c not in drop_obj]
    print(f"  · 투입 변수 : {len(feats)}개 (수치 {len(num)} / 범주 {len(cats)})")
    print(f"  · 제외 변수 : 실적계 {int((dd['운영가용'] == 'N').sum())}개 + 식별·외삽위험 {len(EXCLUDE)}개")
    print(f"  · 범주 변수 : {cats}")

    # ── ② 구간 분할 ────────────────────────────
    cv = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].copy()
    ho = df[df["date"] >= HOLD_START].copy()
    cv = cv.dropna(subset=[t])
    print(f"\n■ ② 평가 구간")
    print(f"  · 교차검증 : {cv['date'].min()} ~ {cv['date'].max()}  {len(cv):,}행")
    print(f"  · 홀드아웃 : {ho['date'].min()} ~ {ho['date'].max()}  {len(ho):,}행")

    # ── ③ 확장창 교차검증 ──────────────────────
    print(f"\n■ ③ 확장창 시계열 교차검증 ({args.folds}분할)")
    days = sorted(cv["date"].unique())
    n_min = min(args.min_train_days, len(days) - args.folds * 30)
    size = max(30, (len(days) - n_min) // args.folds)
    print(f"  · 최소 학습기간 {n_min}일 / 검증창 {size}일")
    print(f"  {'분할':<6}{'학습기간':<26}{'검증기간':<26}{'MAPE':>7}{'RMSE':>8}")
    print("  " + "-" * 73)
    cv_scores, oof = [], []
    for k in range(args.folds):
        tr_days = set(days[: n_min + size * k])
        va_days = set(days[n_min + size * k: n_min + size * (k + 1)])
        if not va_days:
            break
        tr = cv[cv["date"].isin(tr_days)]
        va = cv[cv["date"].isin(va_days)]
        m = make_model(kind, lgb, args.seed, tuned)
        m.fit(tr[feats], tr[t])
        p = pd.Series(m.predict(va[feats]), index=va.index)
        mm = metrics(va[t], p)
        cv_scores.append(mm)
        o = va[["date", "hour_n"]].copy()
        o["y"], o["pred"], o["fold"] = va[t].values, p.values, k + 1
        oof.append(o)
        print(f"  {k + 1:<6}{min(tr_days)}~{max(tr_days):<14}{min(va_days)}~{max(va_days):<14}"
              f"{mm['MAPE']:>7.2f}{mm['RMSE']:>8.2f}")

    cv_mean = {k: float(np.nanmean([s[k] for s in cv_scores]))
               for k in ("MAPE", "sMAPE", "RMSE", "MAE", "R2")}
    print("  " + "-" * 73)
    print(f"  {'평균':<58}{cv_mean['MAPE']:>7.2f}{cv_mean['RMSE']:>8.2f}")

    # ── ④ 최종 학습·홀드아웃 평가 ──────────────
    print("\n■ ④ 최종 학습 및 홀드아웃 평가")
    final = make_model(kind, lgb, args.seed, tuned)
    final.fit(cv[feats], cv[t])
    ho = ho.copy()
    ho["pred"] = final.predict(ho[feats])
    ho_m = metrics(ho[t], ho["pred"])
    print(f"  {'지표':<10}{'MAPE':>8}{'sMAPE':>8}{'RMSE':>9}{'MAE':>8}{'R²':>8}")
    print(f"  {'교차검증':<10}{cv_mean['MAPE']:>8.2f}{cv_mean['sMAPE']:>8.2f}"
          f"{cv_mean['RMSE']:>9.2f}{cv_mean['MAE']:>8.2f}{cv_mean['R2']:>8.3f}")
    print(f"  {'홀드아웃':<10}{ho_m['MAPE']:>8.2f}{ho_m['sMAPE']:>8.2f}"
          f"{ho_m['RMSE']:>9.2f}{ho_m['MAE']:>8.2f}{ho_m['R2']:>8.3f}")

    # ── ⑤ 벤치마크 대비 개선율 ─────────────────
    rows = [{"구간": "교차검증(평균)", "모델": "학습모델", **{k: round(v, 3) for k, v in cv_mean.items()}},
            {"구간": "홀드아웃", "모델": "학습모델", **{k: round(v, 3) for k, v in ho_m.items()}}]
    bpp = base_pred_path(t)
    if os.path.exists(bpp):
        print("\n■ ⑤ 벤치마크 대비 개선율 (홀드아웃)")
        bp = pd.read_csv(bpp, encoding="utf-8-sig")
        if t not in bp.columns:
            sys.exit(f"[중단] 벤치마크 파일 타깃 불일치 → {bpp}\n"
                     f"       python scripts\\step4_baseline.py --target {t} 재실행 필요")
        bp = bp[bp["date"] >= HOLD_START]
        mg = ho[["date", "hour_n", t, "pred"]].merge(bp, on=["date", "hour_n"], how="left")
        bcols = [c for c in bp.columns if c.startswith("B")]
        print(f"  {'구분':<18}{'MAPE':>8}{'RMSE':>9}{'MAPE 개선':>11}")
        print("  " + "-" * 46)
        best_name, best_mape = None, 1e9
        for c in bcols:
            mm = metrics(mg[f"{t}_x"] if f"{t}_x" in mg.columns else mg[t], mg[c])
            if mm["MAPE"] < best_mape:
                best_mape, best_name = mm["MAPE"], c
            rows.append({"구간": "홀드아웃", "모델": c, **{k: round(v, 3) for k, v in mm.items()}})
            print(f"  {c:<18}{mm['MAPE']:>8.2f}{mm['RMSE']:>9.2f}{'':>11}")
        imp = 100 * (best_mape - ho_m["MAPE"]) / best_mape
        print("  " + "-" * 46)
        print(f"  {'학습모델':<18}{ho_m['MAPE']:>8.2f}{ho_m['RMSE']:>9.2f}{imp:>10.1f}%")
        print(f"\n  → 최우수 벤치마크({best_name}, MAPE {best_mape:.2f}%) 대비 "
              f"**{imp:.1f}% 개선**")
        gate = "달성" if ho_m["MAPE"] <= 8.0 else "미달"
        print(f"  → Gate 3 기준(MAPE ≤ 8.0%) : **{gate}** (현 {ho_m['MAPE']:.2f}%)")

    # ── ⑥ 변수 중요도 ──────────────────────────
    print("\n■ ⑥ 변수 중요도 상위 15")
    if hasattr(final, "feature_importances_"):
        fi = pd.DataFrame({"변수": feats, "중요도": final.feature_importances_})
    else:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(final, ho[feats], ho[t], n_repeats=3,
                                   random_state=args.seed, n_jobs=-1)
        fi = pd.DataFrame({"변수": feats, "중요도": r.importances_mean})
    fi["비중(%)"] = 100 * fi["중요도"] / fi["중요도"].sum()
    fi = fi.sort_values("중요도", ascending=False).reset_index(drop=True)
    for i, r in fi.head(15).iterrows():
        print(f"  {i + 1:>2}. {r['변수']:<24}{r['비중(%)']:>7.2f}%")

    # ── ⑦ 오차 분해 ────────────────────────────
    print("\n■ ⑦ 홀드아웃 오차 분해")
    ho["_err"] = ho["pred"] - ho[t]
    ho["_ape"] = (ho["_err"] / ho[t].where(ho[t].abs() >= mape_floor())).abs() * 100
    for label, key in (("일유형별", "daytype"), ("월별", "month")):
        g = ho.groupby(key, observed=True).agg(MAPE=("_ape", "mean"),
                                RMSE=("_err", lambda x: float(np.sqrt((x ** 2).mean()))),
                                N=("_err", "size"))
        print(f"\n  [{label}]")
        for i, r in g.iterrows():
            print(f"  {str(i):<8}MAPE {r['MAPE']:>6.2f}%  RMSE {r['RMSE']:>6.2f}  N {int(r['N']):>6,}")

    # ── ⑧ 저장 ─────────────────────────────────
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    try:
        import joblib
        mp = os.path.join(MODEL_DIR, f"smp_{t}_{kind}.pkl")
        joblib.dump({"model": final, "features": feats, "categorical": cats,
                     "target": t, "trained": ts}, mp)
        print(f"\n  → 모델 저장 {mp}")
    except Exception as e:
        print(f"\n  [경고] 모델 저장 실패 : {type(e).__name__}")

    pd.concat(oof, ignore_index=True).to_csv(
        os.path.join(PROCESSED, f"oof_predictions_{t}.csv"), index=False, encoding="utf-8-sig")
    ho[["date", "hour_n", t, "pred"]].to_csv(
        os.path.join(PROCESSED, f"model_predictions_{t}.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(
        os.path.join(LOG_DIR, f"step4_train_summary_{t}_{ts}.csv"), index=False, encoding="utf-8-sig")
    fi.to_csv(os.path.join(LOG_DIR, f"step4_train_importance_{t}_{ts}.csv"),
              index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f" 홀드아웃 MAPE {ho_m['MAPE']:.2f}% / RMSE {ho_m['RMSE']:.2f}원-kWh")
    print(" ※ 변수 중요도·오차 분해를 근거로 다음 개선 과제 도출")
    print("=" * 78)


if __name__ == "__main__":
    main()
