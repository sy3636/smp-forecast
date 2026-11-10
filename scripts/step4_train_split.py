# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 분리 학습 모델
================================================
목적 : 일유형(평일·휴일) 등 이질 구간을 분리 학습하여 단일 모델의 절충 손실을 제거
배경 : 단일 모델은 휴일(MAPE 17.85→14.37%) 개선 대가로 평일(7.80→8.74%)이 악화됨
       → 구간별 개별 모델로 양측 동시 개선을 도모

입력 : data/processed/smp_features_hourly.csv
       data/processed/feature_dictionary.csv
       data/processed/baseline_predictions_{target}.csv
출력 : ml/models/smp_{target}_split_{구간}.pkl
       data/processed/model_predictions_split_{target}.csv
       logs/step4_split_summary_{일시}.csv

분리 기준(--split)
  daytype : 평일 · 휴일               (기본)
  season  : 봄 · 여름 · 가을 · 겨울
  peak    : 첨두시간(9~21시) · 경부하시간

버전 : v1.0 (2026-08-01)
실행 : python scripts\\step4_train_split.py
       python scripts\\step4_train_split.py --target smp_jeju --split daytype
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import MAPE_FLOOR, mape_floor, set_mape_floor, feat_paths, metrics, TRAIN_START, CV_END, HOLD_START  # noqa: E402
from step4_train import (EXCLUDE, CATEGORICAL, load_model_class,  # noqa: E402
                         make_model, base_pred_path)

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


def make_split_key(df, mode):
    """분리 기준별 구간 라벨 생성."""
    if mode == "daytype":
        return df["daytype"].astype(str)
    if mode == "season":
        return df["season"].astype(str)
    if mode == "peak":
        return np.where(df["hour_n"].between(9, 21), "첨두", "경부하")
    sys.exit(f"[중단] 미지원 분리 기준 : {mode}")


def fit_predict(train, valid, feats, t, kind, lgb, seed):
    m = make_model(kind, lgb, seed)
    m.fit(train[feats], train[t])
    return m, pd.Series(m.predict(valid[feats]), index=valid.index)


def main():
    ap = argparse.ArgumentParser(description="step4 분리 학습 모델")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--split", default="daytype", choices=["daytype", "season", "peak"])
    ap.add_argument("--folds", type=int, default=5)
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    kind, lgb = load_model_class()

    print("=" * 78)
    print(" step4 분리 학습 모델 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 타깃 : {t} / 분리기준 : {args.split} / 엔진 : {kind}")
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

    df["_grp"] = make_split_key(df, args.split)
    groups = sorted(pd.Series(df["_grp"]).dropna().unique())
    print(f"\n■ ① 분리 구간 : {len(groups)}개 — {groups}")
    print(f"  · 투입 변수 : {len(feats)}개")

    cv = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].dropna(subset=[t]).copy()
    ho = df[df["date"] >= HOLD_START].copy()
    for g in groups:
        print(f"  · {g:<8} 학습 {int((cv['_grp'] == g).sum()):>7,}행 / "
              f"홀드아웃 {int((ho['_grp'] == g).sum()):>6,}행")

    # ── ② 교차검증 : 단일 vs 분리 ──────────────
    print(f"\n■ ② 확장창 교차검증 ({args.folds}분할) — 단일 vs 분리")
    days = sorted(cv["date"].unique())
    n_min = min(args.min_train_days, len(days) - args.folds * 30)
    size = max(30, (len(days) - n_min) // args.folds)

    cv_single, cv_split = [], []
    for k in range(args.folds):
        tr_d = set(days[: n_min + size * k])
        va_d = set(days[n_min + size * k: n_min + size * (k + 1)])
        if not va_d:
            break
        tr, va = cv[cv["date"].isin(tr_d)], cv[cv["date"].isin(va_d)]

        _, p1 = fit_predict(tr, va, feats, t, kind, lgb, args.seed)
        cv_single.append(metrics(va[t], p1))

        p2 = pd.Series(index=va.index, dtype=float)
        for g in groups:
            trg, vag = tr[tr["_grp"] == g], va[va["_grp"] == g]
            if len(trg) < 200 or vag.empty:
                p2.loc[vag.index] = p1.loc[vag.index] if not vag.empty else None
                continue
            _, pg = fit_predict(trg, vag, feats, t, kind, lgb, args.seed)
            p2.loc[vag.index] = pg
        cv_split.append(metrics(va[t], p2))
        print(f"  분할 {k + 1} : 단일 {cv_single[-1]['MAPE']:>6.2f}%  →  "
              f"분리 {cv_split[-1]['MAPE']:>6.2f}%")

    def avg(lst, key):
        return float(np.nanmean([x[key] for x in lst]))

    print("  " + "-" * 50)
    print(f"  평균    : 단일 {avg(cv_single, 'MAPE'):>6.2f}%  →  "
          f"분리 {avg(cv_split, 'MAPE'):>6.2f}%")

    # ── ③ 홀드아웃 : 단일 vs 분리 ──────────────
    print(f"\n■ ③ 홀드아웃 평가 ({ho['date'].min()} ~ {ho['date'].max()})")
    ms, ps = fit_predict(cv, ho, feats, t, kind, lgb, args.seed)
    ho = ho.copy()
    ho["pred_single"] = ps
    ho["pred_split"] = np.nan
    models = {}
    for g in groups:
        cg, hg = cv[cv["_grp"] == g], ho[ho["_grp"] == g]
        if len(cg) < 200 or hg.empty:
            ho.loc[hg.index, "pred_split"] = ho.loc[hg.index, "pred_single"]
            continue
        m, pg = fit_predict(cg, hg, feats, t, kind, lgb, args.seed)
        models[g] = m
        ho.loc[hg.index, "pred_split"] = pg

    m_single = metrics(ho[t], ho["pred_single"])
    m_split = metrics(ho[t], ho["pred_split"])
    print(f"  {'구분':<10}{'MAPE':>8}{'sMAPE':>8}{'RMSE':>9}{'MAE':>8}{'R²':>8}")
    print(f"  {'단일모델':<10}{m_single['MAPE']:>8.2f}{m_single['sMAPE']:>8.2f}"
          f"{m_single['RMSE']:>9.2f}{m_single['MAE']:>8.2f}{m_single['R2']:>8.3f}")
    print(f"  {'분리모델':<10}{m_split['MAPE']:>8.2f}{m_split['sMAPE']:>8.2f}"
          f"{m_split['RMSE']:>9.2f}{m_split['MAE']:>8.2f}{m_split['R2']:>8.3f}")
    gain = m_single["MAPE"] - m_split["MAPE"]
    print(f"\n  → MAPE {gain:+.2f}%p "
          f"({'개선' if gain > 0 else '악화'}, 상대 {100 * gain / m_single['MAPE']:+.1f}%)")

    # ── ④ 구간별 성능 ──────────────────────────
    print("\n■ ④ 구간별 성능 (홀드아웃)")
    print(f"  {'구간':<10}{'단일':>10}{'분리':>10}{'개선(%p)':>11}{'N':>8}")
    rows = []
    for g in groups:
        s = ho[ho["_grp"] == g]
        if s.empty:
            continue
        a, b = metrics(s[t], s["pred_single"]), metrics(s[t], s["pred_split"])
        print(f"  {g:<10}{a['MAPE']:>10.2f}{b['MAPE']:>10.2f}"
              f"{a['MAPE'] - b['MAPE']:>11.2f}{b['N']:>8,}")
        rows.append({"구간": g, "단일_MAPE": round(a["MAPE"], 2),
                     "분리_MAPE": round(b["MAPE"], 2),
                     "개선": round(a["MAPE"] - b["MAPE"], 2), "N": b["N"]})

    # ── ⑤ 벤치마크 대비 ────────────────────────
    bpp = base_pred_path(t)
    best_b = None
    if os.path.exists(bpp):
        bp = pd.read_csv(bpp, encoding="utf-8-sig")
        bp = bp[bp["date"] >= HOLD_START]
        mg = ho[["date", "hour_n", t]].merge(bp, on=["date", "hour_n"], how="left")
        y = mg[f"{t}_x"] if f"{t}_x" in mg.columns else mg[t]
        cand = {c: metrics(y, mg[c])["MAPE"] for c in bp.columns if c.startswith("B")}
        best_b = min(cand, key=cand.get)
        print(f"\n■ ⑤ 벤치마크 대비")
        print(f"  · 최우수 벤치마크 : {best_b} {cand[best_b]:.2f}%")
        print(f"  · 단일모델        : {m_single['MAPE']:.2f}% "
              f"({100 * (cand[best_b] - m_single['MAPE']) / cand[best_b]:+.1f}%)")
        print(f"  · 분리모델        : {m_split['MAPE']:.2f}% "
              f"({100 * (cand[best_b] - m_split['MAPE']) / cand[best_b]:+.1f}%)")
        gate = "달성" if m_split["MAPE"] <= 8.0 else "미달"
        print(f"  · Gate 3(≤8.0%)  : {gate}")

    # ── ⑥ 저장 ─────────────────────────────────
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    try:
        import joblib
        for g, m in models.items():
            joblib.dump({"model": m, "features": feats, "target": t,
                         "split": args.split, "group": g, "trained": ts},
                        os.path.join(MODEL_DIR, f"smp_{t}_split_{g}.pkl"))
        print(f"\n  → 모델 {len(models)}건 저장 : {MODEL_DIR}")
    except Exception as e:
        print(f"\n  [경고] 모델 저장 실패 : {type(e).__name__}")

    ho[["date", "hour_n", "_grp", t, "pred_single", "pred_split"]].to_csv(
        os.path.join(PROCESSED, f"model_predictions_split_{t}.csv"),
        index=False, encoding="utf-8-sig")

    summ = [{"구간": "홀드아웃", "모델": "단일", **{k: round(v, 3) for k, v in m_single.items()}},
            {"구간": "홀드아웃", "모델": f"분리({args.split})",
             **{k: round(v, 3) for k, v in m_split.items()}}]
    pd.DataFrame(summ + rows).to_csv(
        os.path.join(LOG_DIR, f"step4_split_summary_{t}_{ts}.csv"),
        index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f" 결론 : 분리({args.split}) MAPE {m_split['MAPE']:.2f}% "
          f"/ 단일 대비 {gain:+.2f}%p")
    print("=" * 78)


if __name__ == "__main__":
    main()
