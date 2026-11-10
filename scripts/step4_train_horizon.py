# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 지평별 운영 모델 학습
================================================
배경 : SMP 확정값은 D-1 15~23시 사이에 공개됨(2026.8.12 실측)
       → 공개 이전 시각(14~15시)에 예측하면 **D+1도 실질 예측 가치**를 가짐
       그 시점에는 D+1 수요예측도 미확보이므로 **수요예측 변수 미사용** 모델이 필요

목적 : D+1 ~ D+6 각 지평별 운영 모델을 학습·저장하여 주간 예측 체계를 구성

설계
  ① 타깃 시프트 : y_H(t) = y(t + 24×(H-1))
     · 피처는 lag24 이상만 사용 → 타깃 시프트로 상대 시차가 24×H로 확대
     · 학습–운영 정합성(train-serving skew) 자동 확보
  ② 수요예측 변수 **전 지평 미사용** — 예측 시각(14~15시)에 D+1분도 미공개
  ③ 지평별 독립 모델(direct multi-horizon) — 오차 누적 없음

산출 모델
  ml/models/smp_{타깃}_h{H}.pkl   (H = 1~6)
    · model / features / categorical / horizon / scenario / trained_at 포함

입력 : data/processed/smp_features_{타깃}.csv
       data/processed/feature_dictionary_{타깃}.csv
출력 : ml/models/smp_{타깃}_h{H}.pkl
       logs/step4_train_horizon_{타깃}_{일시}.csv

버전 : v1.1 (2026-09-03) — 주지표 MAE·rMAE 산출 추가
  · 배경 : 4지표 체계(PRD v2.13) 전환 후에도 지평별 산출은 MAPE·RMSE에 한정
  · 조치 : 지평별 MAE·rMAE(naive-1 기준)를 병행 산출·저장
  · 용도 : 주간(D+2~D+6) MAE 목표 확정 근거 — 목표오차 확정서 v1.0 Action Item 1
실행 : python scripts\\step4_train_horizon.py --target smp_land
       python scripts\\step4_train_horizon.py --target smp_land --max-h 6
"""

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

try:
    import joblib
except ImportError:
    sys.exit("[중단] joblib 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TRAIN_START = "2021-01-01"
CV_END = "2025-12-31"
HOLD_START = "2026-01-01"
MAPE_FLOOR = 20.0
EXCLUDE = {"date", "hour", "daytype", "year"}
CATEGORICAL = ["month", "dow", "hour_n"]

REPORT = []


def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def rep(구분, 항목, 값, 비고=""):
    REPORT.append({"구분": 구분, "항목": 항목, "값": str(값), "비고": 비고})
    print(f"  · {항목} : {값}" + (f"  ({비고})" if 비고 else ""))


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    if not m.any():
        return None
    return float(np.mean(np.abs(p[m] - a[m]) / np.clip(a[m], MAPE_FLOOR, None)) * 100)


def mae(a, p):
    """평균절대오차(원/kWh) — 주지표 ①"""
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    return float(np.mean(np.abs(p[m] - a[m]))) if m.any() else None


def rmse(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    return float(np.sqrt(np.mean((p[m] - a[m]) ** 2))) if m.any() else None


def load_tuned(t):
    """튜닝 파라미터 적용 — 존재 시 채택, 없으면 기본값"""
    p = os.path.join(BASE_DIR, "config", f"model_params_{t}.json")
    if os.path.exists(p):
        try:
            import json
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return d.get("params", d), os.path.basename(p)
        except Exception:
            pass
    return None, "기본값"


def make_model(params=None, seed=42):
    """LightGBM 우선, 미설치 시 scikit-learn 대체"""
    try:
        import lightgbm as lgb
        p = dict(params or {})
        p.setdefault("n_estimators", 600)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("num_leaves", 63)
        p.setdefault("min_child_samples", 30)
        p["objective"] = p.pop("objective", "regression")
        for k in ("n_estimators", "num_leaves", "min_child_samples"):
            if k in p:
                p[k] = int(p[k])
        return lgb.LGBMRegressor(random_state=seed, verbose=-1, **p), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=seed), "sklearn"


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="지평별 운영 모델 학습")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--max-h", type=int, default=6, help="최대 예측 지평(일)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-demand", action="store_true",
                    help="수요예측 변수 사용(기본 미사용 — 예측 시각에 미확보)")
    ap.add_argument("--variant", default="",
                    help="실증용 피처셋 접미사(예: _fuel). 운영 산출물과 분리")
    ap.add_argument("--no-save", action="store_true",
                    help="모델 미저장 — 성능 비교(A/B) 전용 실행")
    ap.add_argument("--drop-prefix", default="",
                    help="투입 제외 변수 접두어(쉼표 구분) — 정보 중복 검증용")
    ap.add_argument("--train-start", default=None,
                    help="학습 시작일 재지정(YYYY-MM-DD) — 실증 조건 통제용")
    args = ap.parse_args()
    t = args.target

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"

    print("=" * 78)
    print(" step4 지평별 운영 모델 학습 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    v = args.variant
    src = os.path.join(PROCESSED, f"smp_features_{t}{v}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}{v}.csv")
    for p in (src, dic):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}\n"
                     f"       python scripts\\step3_features.py --target {t} 선행 필요")

    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(dic, encoding="utf-8-sig")

    print("\n■ ① 변수 선정")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    feats = [c for c in feats if is_num(df[c])]
    demand = [c for c in feats if c.startswith("demand_fcst")]
    if not args.use_demand:
        feats = [c for c in feats if c not in demand]
    if args.drop_prefix:
        pre = tuple(x.strip() for x in args.drop_prefix.split(",") if x.strip())
        n0 = len(feats)
        feats = [c for c in feats if not c.startswith(pre)]
        rep("변수", "제외 변수", f"{n0 - len(feats)}개", f"접두어 {pre}")
    cats = [c for c in CATEGORICAL if c in feats]
    for c in cats:
        df[c] = df[c].astype("category")

    scen = "수요예측 사용" if args.use_demand else "수요예측 미사용"
    rep("변수", "투입 변수", f"{len(feats)}개", f"범주형 {len(cats)}개")
    rep("변수", "수요예측 계열", f"{len(demand)}개",
        "사용" if args.use_demand else "제외 — 예측 시각(14~15시)에 미확보")
    params, psrc = load_tuned(t)
    rep("설정", "파라미터", psrc)
    rep("설정", "시나리오", scen)

    # ── 지평별 학습 ────────────────────────────
    print(f"\n■ ② 지평별 학습·저장 ({args.max_h}단계)")
    print(f"  {'지평':<7}{'검증행':>8}{'MAE':>8}{'rMAE':>8}{'MAPE':>9}{'RMSE':>8}"
          f"{'지속성':>9}{'개선율':>9}   저장")
    print("  " + "-" * 78)

    rows = []
    for h in range(1, args.max_h + 1):
        shift = 24 * (h - 1)
        d = df.copy()
        d["_y"] = d[t].shift(-shift)
        d["_naive"] = (d["smp_lag_d1"] if "smp_lag_d1" in d.columns
                       else d[t].shift(24))

        _ts = args.train_start or TRAIN_START
        tr = d[(d["date"] >= _ts) & (d["date"] <= CV_END)].dropna(subset=["_y"])
        ho = d[d["date"] >= HOLD_START].dropna(subset=["_y"])
        if len(tr) < 5000 or len(ho) < 500:
            print(f"  D+{h:<5} 표본 부족 — 건너뜀")
            continue

        m, kind = make_model(params, args.seed)
        m.fit(tr[feats], tr["_y"])
        pred = m.predict(ho[feats])
        mp, rm = mape(ho["_y"], pred), rmse(ho["_y"], pred)
        nv = mape(ho["_y"], ho["_naive"])
        imp = (nv - mp) / nv * 100 if (nv and mp) else None
        # 주지표 — MAE·rMAE(naive-1 = 전일 동시간 실적, 정의 변경 금지)
        ma = mae(ho["_y"], pred)
        ma_nv = mae(ho["_y"], ho["_naive"])
        rma = (ma / ma_nv) if (ma and ma_nv) else None

        # 전 구간 재학습 후 저장 — 운영 모델은 최신 정보 반영
        #  · --no-save 지정 시 생략(A/B 비교 실행에서 운영 모델 보호)
        full = d.dropna(subset=["_y"])
        if args.no_save:
            print(f"  D+{h:<5}{len(ho):>8,}{ma:>8.2f}{rma:>8.3f}{mp:>8.2f}%"
                  f"{rm:>8.2f}{nv:>8.2f}%{imp:>8.1f}%   (미저장)")
            rows.append({"지평": f"D+{h}", "H": h, "변수수": len(feats),
                         "학습행": len(tr), "검증행": len(ho),
                         "MAE": round(ma, 2), "rMAE": round(rma, 3),
                         "naiveMAE": round(ma_nv, 2),
                         "MAPE": round(mp, 2), "RMSE": round(rm, 2),
                         "지속성MAPE": round(nv, 2),
                         "개선율": round(imp, 1) if imp else None,
                         "모델파일": "-"})
            continue
        mf, _ = make_model(params, args.seed)
        mf.fit(full[feats], full["_y"])

        mpth = os.path.join(MODEL_DIR, f"smp_{t}{v}_h{h}.pkl")
        tmp = mpth + ".tmp"
        joblib.dump({"model": mf, "features": feats, "categorical": cats,
                     "horizon": h, "scenario": scen, "engine": kind,
                     "trained_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                     "holdout_mape": round(mp, 3) if mp else None,
                     "holdout_rmse": round(rm, 3) if rm else None,
                     "holdout_mae": round(ma, 3) if ma else None,
                     "holdout_rmae": round(rma, 4) if rma else None}, tmp)
        os.replace(tmp, mpth)              # 원자적 저장 — 중단 시 손상 방지

        print(f"  D+{h:<5}{len(ho):>8,}{ma:>8.2f}{rma:>8.3f}{mp:>8.2f}%"
              f"{rm:>8.2f}{nv:>8.2f}%{imp:>8.1f}%   {os.path.basename(mpth)}")
        rows.append({"지평": f"D+{h}", "H": h, "변수수": len(feats),
                     "학습행": len(full), "검증행": len(ho),
                     "MAE": round(ma, 2), "rMAE": round(rma, 3),
                     "naiveMAE": round(ma_nv, 2),
                     "MAPE": round(mp, 2), "RMSE": round(rm, 2),
                     "지속성MAPE": round(nv, 2),
                     "개선율": round(imp, 1) if imp else None,
                     "모델파일": os.path.basename(mpth)})
    print("  " + "-" * 78)

    if not rows:
        sys.exit("[중단] 학습 가능한 지평이 없음")

    r = pd.DataFrame(rows)
    rep("결과", "저장 모델", f"{len(r)}종",
        f"D+1 ~ D+{int(r['H'].max())}")
    rep("결과", "MAE 범위",
        f"{r['MAE'].min():.2f} ~ {r['MAE'].max():.2f}원/kWh", "주지표 ①")
    rep("결과", "rMAE 범위",
        f"{r['rMAE'].min():.3f} ~ {r['rMAE'].max():.3f}",
        "주지표 ② — naive-1(전일 동시간) 기준")
    rep("결과", "MAPE 범위",
        f"{r['MAPE'].min():.2f}% ~ {r['MAPE'].max():.2f}%")
    rep("결과", "개선율 범위",
        f"{r['개선율'].min():.1f}% ~ {r['개선율'].max():.1f}%",
        "지속성 모형 대비")

    lp = os.path.join(LOG_DIR, f"step4_train_horizon_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["구분", "항목", "값", "비고"])
        for x in REPORT:
            w.writerow([x["구분"], x["항목"], x["값"], x["비고"]])
        w.writerow([])
        w.writerow(list(r.columns))
        for _, x in r.iterrows():
            w.writerow(list(x.values))
    print(f"\n  → 검증보고 {lp}")

    print("=" * 78)
    print(f" 학습 완료 : {len(r)}개 지평 모델 / {scen}")
    print(f" 다음 단계 : python scripts\\step6_weekly_predict.py --target {t}")
    print("=" * 78)


if __name__ == "__main__":
    main()
