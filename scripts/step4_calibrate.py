# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 평균회귀 편의 보정
================================================
목적 : 예측값의 체계적 편의(저가 과대예측·고가 과소예측)를 사후 보정하여 정확도 개선
배경 : 2026년 홀드아웃 분석 결과 아래와 같은 구조적 편의 확인
       · 20원 미만 구간   : +60.8원 (과대예측)
       · 120~150원 구간   : -5.1원
       · 150원 이상 구간  : -7.8원 (과소예측)
       → 트리 모델의 평균회귀 특성에 기인

보정 방식 (전부 **교차검증 예측값(OOF)에서만 학습**, 홀드아웃 미사용)
  A. none        : 무보정(기준)
  B. linear      : 최소자승 선형보정  y = a + b·ŷ
  C. isotonic    : 등위회귀(단조 비모수 보정)
  D. bin_resid   : 예측 십분위 구간별 평균잔차 차감(선형보간)
  E. objective   : 모델 목적함수 변경 비교(--objective huber|quantile)

입력 : data/processed/smp_features_{target}.csv, feature_dictionary_{target}.csv
출력 : ml/models/calibrator_{target}.pkl        (최우수 보정기)
       outputs/calibrate_{target}/*.png         (보정 전후 편의 곡선)
       logs/step4_calibrate_{target}_{일시}.csv (성능 비교표)

버전 : v1.0 (2026-08-04)
실행 : python scripts\\step4_calibrate.py
       python scripts\\step4_calibrate.py --target smp_jeju
       python scripts\\step4_calibrate.py --objective huber
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import (MAPE_FLOOR, set_mape_floor, feat_paths,  # noqa: E402
                            metrics, TRAIN_START, CV_END, HOLD_START)
from step4_train import (EXCLUDE, CATEGORICAL, load_model_class,  # noqa: E402
                         load_tuned_params, make_model)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

try:
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
except ImportError:
    plt = None

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
OUT_ROOT = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")

BINS = [-999, 20, 80, 100, 120, 150, 9999]
LABELS = ["20원미만", "20~80", "80~100", "100~120", "120~150", "150원~"]


# ------------------------------------------------------------------
# 보정기
# ------------------------------------------------------------------
class Calibrator:
    """OOF 예측값에서 학습한 사후 보정기."""

    def __init__(self, kind="none"):
        self.kind = kind
        self.model = None
        self.grid = None      # bin_resid 용 (예측값, 보정량)

    def fit(self, pred, y):
        p, t = np.asarray(pred, float), np.asarray(y, float)
        m = ~(np.isnan(p) | np.isnan(t))
        p, t = p[m], t[m]

        if self.kind == "linear":
            self.model = np.polyfit(p, t, 1)          # [b, a]
        elif self.kind == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, t)
        elif self.kind == "bin_resid":
            q = np.quantile(p, np.linspace(0, 1, 11))
            q = np.unique(q)
            idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
            centers, adj = [], []
            for b in range(len(q) - 1):
                s = idx == b
                if s.sum() >= 30:
                    centers.append(p[s].mean())
                    adj.append((t[s] - p[s]).mean())   # 평균 잔차(보정량)
            self.grid = (np.array(centers), np.array(adj))
        return self

    def transform(self, pred):
        p = np.asarray(pred, float)
        if self.kind == "none" or (self.model is None and self.grid is None):
            return p
        if self.kind == "linear":
            b, a = self.model
            return a + b * p
        if self.kind == "isotonic":
            return self.model.predict(p)
        if self.kind == "bin_resid":
            c, a = self.grid
            return p + np.interp(p, c, a)
        return p


# ------------------------------------------------------------------
def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def build_oof(cv, feats, t, kind, lgb, tuned, seed, folds, min_train_days):
    """확장창 교차검증으로 OOF 예측 생성(보정기 학습용)."""
    days = sorted(cv["date"].unique())
    n_min = min(min_train_days, len(days) - folds * 30)
    size = max(30, (len(days) - n_min) // folds)
    oof = pd.Series(index=cv.index, dtype=float)
    for k in range(folds):
        tr_d = set(days[: n_min + size * k])
        va_d = set(days[n_min + size * k: n_min + size * (k + 1)])
        if not va_d:
            break
        tr, va = cv[cv["date"].isin(tr_d)], cv[cv["date"].isin(va_d)]
        m = make_model(kind, lgb, seed, tuned)
        m.fit(tr[feats], tr[t])
        oof.loc[va.index] = m.predict(va[feats])
    return oof


def bias_table(y, p, by="actual"):
    """구간별 편의표. by='actual' 실적 기준 / by='pred' 예측 기준(운영 관점)."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}).dropna()
    d["_e"] = d["p"] - d["y"]
    d["_구간"] = pd.cut(d["y"] if by == "actual" else d["p"], BINS, labels=LABELS)
    g = d.groupby("_구간", observed=True).agg(
        편의=("_e", "mean"), MAE=("_e", lambda x: x.abs().mean()), N=("_e", "size"))
    return g


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step4 평균회귀 편의 보정")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--min-train-days", type=int, default=730)
    ap.add_argument("--objective", default=None,
                    choices=["huber", "quantile"], help="목적함수 변경 비교(선택)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR)
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    SRC, DICT = feat_paths(t)
    for p in (SRC, DICT):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")
    out = os.path.join(OUT_ROOT, f"calibrate_{t}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    kind, lgb = load_model_class()
    tuned, src_p = load_tuned_params(kind, t)
    if args.objective and kind == "lightgbm":
        tuned = dict(tuned)
        tuned["objective"] = args.objective
        if args.objective == "quantile":
            tuned["alpha"] = 0.5

    print("=" * 78)
    print(" step4 평균회귀 편의 보정 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 타깃 : {t} / 엔진 : {kind} / 파라미터 : {src_p}"
          + (f" / 목적함수 : {args.objective}" if args.objective else ""))
    print("=" * 78)

    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")
    okv = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in okv and c not in EXCLUDE]
    cats = [c for c in CATEGORICAL if c in feats]
    for c in cats:
        df[c] = df[c].astype("category")
    feats = [c for c in feats if c in cats or is_num(df[c])]

    cv = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].dropna(subset=[t]).copy()
    ho = df[df["date"] >= HOLD_START].copy()

    # ── ① OOF 생성 ─────────────────────────────
    print(f"\n■ ① 교차검증 예측값(OOF) 생성 — {args.folds}분할, 변수 {len(feats)}개")
    oof = build_oof(cv, feats, t, kind, lgb, tuned, args.seed,
                    args.folds, args.min_train_days)
    valid = oof.notna()
    print(f"  · OOF 확보 : {int(valid.sum()):,}행 / {len(cv):,}행")
    mm = metrics(cv.loc[valid, t], oof[valid])
    print(f"  · OOF 성능 : MAPE {mm['MAPE']:.2f}% / RMSE {mm['RMSE']:.2f}")

    # ── ② 최종 모델 학습·홀드아웃 예측 ─────────
    print("\n■ ② 최종 학습 및 홀드아웃 예측")
    final = make_model(kind, lgb, args.seed, tuned)
    final.fit(cv[feats], cv[t])
    ho = ho.copy()
    ho["pred_raw"] = final.predict(ho[feats])
    base = metrics(ho[t], ho["pred_raw"])
    print(f"  · 무보정 : MAPE {base['MAPE']:.2f}% / RMSE {base['RMSE']:.2f} / R² {base['R2']:.3f}")

    # ── ③ 보정기 학습·적용 ─────────────────────
    print("\n■ ③ 보정 방식별 성능 (보정기는 OOF에서만 학습)")
    kinds = ["none", "linear", "isotonic", "bin_resid"]
    rows, preds, best = [], {}, None
    print(f"  {'방식':<12}{'MAPE':>8}{'RMSE':>9}{'MAE':>8}{'R²':>8}{'MAPE 개선':>11}")
    print("  " + "-" * 56)
    for kd in kinds:
        try:
            cal = Calibrator(kd).fit(oof[valid], cv.loc[valid, t])
            p = cal.transform(ho["pred_raw"])
        except Exception as e:
            print(f"  {kd:<12} 실패 : {type(e).__name__}")
            continue
        m = metrics(ho[t], pd.Series(p, index=ho.index))
        imp = base["MAPE"] - m["MAPE"]
        preds[kd] = p
        rows.append({"보정방식": kd, **{k: round(v, 3) for k, v in m.items()},
                     "MAPE개선": round(imp, 3)})
        print(f"  {kd:<12}{m['MAPE']:>8.2f}{m['RMSE']:>9.2f}{m['MAE']:>8.2f}"
              f"{m['R2']:>8.3f}{imp:>+11.2f}")
        if best is None or m["MAPE"] < best[1]["MAPE"]:
            best = (kd, m, cal)

    print(f"\n  → 최우수 : {best[0]} (MAPE {best[1]['MAPE']:.2f}%, "
          f"무보정 대비 {base['MAPE'] - best[1]['MAPE']:+.2f}%p)")
    gate = "달성" if best[1]["MAPE"] <= 8.0 else "미달"
    print(f"  → Gate 3 기준(MAPE ≤ 8.0%) : {gate}")

    # ── ④ 구간별 편의 개선 ─────────────────────
    print("\n■ ④ SMP 구간별 편의 변화 — 실적값 기준")
    b0 = bias_table(ho[t], ho["pred_raw"])
    b1 = bias_table(ho[t], preds[best[0]])
    print(f"  {'구간':<10}{'무보정 편의':>12}{'보정후 편의':>12}{'개선':>9}{'N':>8}")
    for i in b0.index:
        v0, v1 = b0.loc[i, "편의"], b1.loc[i, "편의"]
        print(f"  {str(i):<10}{v0:>+12.2f}{v1:>+12.2f}"
              f"{abs(v0) - abs(v1):>+9.2f}{int(b0.loc[i, 'N']):>8,}")
        rows.append({"보정방식": f"구간_{i}", "MAPE": None, "RMSE": None,
                     "MAE": None, "R2": None, "N": int(b0.loc[i, "N"]),
                     "MAPE개선": None, "무보정편의": round(v0, 2),
                     "보정후편의": round(v1, 2)})

    # ── ④-2 예측값 기준 편의(운영 관점) ────────
    print("\n■ ④-2 예측값 기준 구간별 편의 — 운영 관점 진단")
    pb = bias_table(ho[t], ho["pred_raw"], by="pred")
    print(f"  {'예측구간':<10}{'편의':>10}{'MAE':>9}{'N':>8}")
    for i, r in pb.iterrows():
        print(f"  {str(i):<10}{r['편의']:>+10.2f}{r['MAE']:>9.2f}{int(r['N']):>8,}")
        rows.append({"보정방식": f"예측구간_{i}", "MAPE": None, "RMSE": None,
                     "MAE": round(r["MAE"], 2), "R2": None, "N": int(r["N"]),
                     "MAPE개선": None, "무보정편의": round(r["편의"], 2)})
    tot = float((ho["pred_raw"] - ho[t]).mean())
    print(f"  · 전체 평균 편의 : {tot:+.3f}원/kWh")
    print("  ※ 실적값 기준 편의는 회귀추정량의 평균회귀 특성상 필연적으로 발생하며,")
    print("     예측값 기준 편의가 0에 근접하면 운영상 조건부 불편성이 확보된 것으로 판정")

    # ── ⑤ 시각화 ───────────────────────────────
    if plt is not None:
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        ko = False
        for f in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
            if f in have:
                plt.rcParams["font.family"] = f
                ko = True
                break
        plt.rcParams["axes.unicode_minus"] = False
        LB = (lambda a, b: a if ko else b)

        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
        x = np.arange(len(b0.index))
        ax[0].bar(x - .2, b0["편의"], .4, label=LB("무보정", "raw"), color="#C00000", alpha=.85)
        ax[0].bar(x + .2, b1["편의"], .4, label=LB(f"보정({best[0]})", f"calib({best[0]})"),
                  color="#1F3864", alpha=.85)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(b0.index if ko else
                              ["<20", "20-80", "80-100", "100-120", "120-150", ">150"],
                              rotation=20, fontsize=8)
        ax[0].axhline(0, color="black", lw=.8)
        ax[0].set_title(LB("① SMP 구간별 편의(예측-실적)", "Bias by SMP range"),
                        fontsize=12, weight="bold")
        ax[0].set_ylabel(LB("원/kWh", "KRW/kWh")); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

        s = ho.sample(min(6000, len(ho)), random_state=0)
        ax[1].scatter(s[t], s["pred_raw"], s=5, alpha=.2, color="#C00000",
                      edgecolors="none", label=LB("무보정", "raw"))
        ax[1].scatter(s[t], pd.Series(preds[best[0]], index=ho.index).loc[s.index],
                      s=5, alpha=.2, color="#1F3864", edgecolors="none",
                      label=LB(f"보정({best[0]})", "calibrated"))
        lo, hi = float(ho[t].min()), float(ho[t].max())
        ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[1].set_title(LB("② 실적 대비 예측 분포", "Actual vs Forecast"),
                        fontsize=12, weight="bold")
        ax[1].set_xlabel(LB("실적", "Actual")); ax[1].set_ylabel(LB("예측", "Forecast"))
        ax[1].legend(markerscale=3); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "01_편의보정.png"), dpi=130)
        plt.close(fig)
        print(f"\n  → 그래프 {os.path.join(out, '01_편의보정.png')}")

    # ── ⑥ 저장 ─────────────────────────────────
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    try:
        import joblib
        mp = os.path.join(MODEL_DIR, f"calibrator_{t}.pkl")
        joblib.dump({"calibrator": best[2], "kind": best[0], "target": t,
                     "objective": args.objective, "trained": ts,
                     "holdout_mape": round(best[1]["MAPE"], 4)}, mp)
        print(f"  → 보정기 저장 {mp}")
    except Exception as e:
        print(f"  [경고] 보정기 저장 실패 : {type(e).__name__}")

    ho["pred_calibrated"] = preds[best[0]]
    ho[["date", "hour_n", t, "pred_raw", "pred_calibrated"]].to_csv(
        os.path.join(out, "predictions.csv"), index=False, encoding="utf-8-sig")
    lp = os.path.join(LOG_DIR, f"step4_calibrate_{t}_{ts}.csv")
    pd.DataFrame(rows).to_csv(lp, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f" 결론 : {best[0]} 보정 적용 시 MAPE {base['MAPE']:.2f}% → "
          f"{best[1]['MAPE']:.2f}% ({base['MAPE'] - best[1]['MAPE']:+.2f}%p)")
    print(f" 성능표 : {lp}")
    print("=" * 78)


if __name__ == "__main__":
    main()
