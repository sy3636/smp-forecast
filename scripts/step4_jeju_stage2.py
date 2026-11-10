# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 제주 2단계 예측(육지 예측값 연동)
================================================
목적 : 제주 Gate 3(MAPE ≤ 8.0%) 미달 해소
배경 : 제주는 HVDC 연계로 육지 계통과 통합되어 가는 추세
       · 육지·제주 상관   : 2022년 0.17~0.28 → 2026년 0.77~1.00
       · 1원 이내 동일비율 : 2023년 77.2% → 2026년 90.9%
       · 스프레드 표준편차 : 2022년 70~77원 → 2026년 15.6원
       → 육지 SMP 정보가 제주 예측의 최대 설명변수로 판단

설계 — 2단계 구조
  [1단계] 육지 모델 예측(기 구축, 홀드아웃 MAPE 7.18%)
      ↓
  [2단계] 육지 예측값 + 스프레드 구조를 제주 모델에 투입

누설 방지
  · 교차검증 구간 : 육지 OOF 예측값 사용(해당 시점 정보만으로 생성)
  · 홀드아웃 구간 : 육지 최종모델 예측값 사용
  · 스프레드 변수 : lag24 이상만 사용(당일 실적 미참조)

학습 구간
  · 육지 OOF 보유 구간(2023-02~)으로 한정 — 구조 변화 시점과 일치하여 타당

입력 : data/processed/smp_features_smp_jeju.csv
       data/processed/oof_predictions_smp_land.csv     (교차검증 구간)
       data/processed/model_predictions_smp_land.csv   (홀드아웃 구간)
출력 : ml/models/smp_smp_jeju_stage2.pkl
       logs/step4_jeju_stage2_{일시}.csv
       outputs/jeju_stage2/*.png

버전 : v1.0 (2026-08-06)
실행 : python scripts\\step4_jeju_stage2.py
       python scripts\\step4_jeju_stage2.py --folds 3
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import (MAPE_FLOOR, set_mape_floor, feat_paths,  # noqa: E402
                            metrics, HOLD_START)
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
OUT_DIR = os.path.join(BASE_DIR, "outputs", "jeju_stage2")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TARGET = "smp_jeju"
LAND = "smp_land"
C1, C2 = "#1F3864", "#C00000"


def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def rep(항목, 값, 비고=""):
    print(f"  · {항목} : {값}" + (f"  ({비고})" if 비고 else ""))


# ------------------------------------------------------------------
def load_land_pred():
    """육지 예측값 결합 — 교차검증 구간은 OOF, 홀드아웃 구간은 최종모델 예측."""
    po = os.path.join(PROCESSED, "oof_predictions_smp_land.csv")
    ph = os.path.join(PROCESSED, "model_predictions_smp_land.csv")
    for p in (po, ph):
        if not os.path.exists(p):
            sys.exit(f"[중단] 육지 예측값 없음 → {p}\n"
                     f"       python scripts\\step4_train.py --target smp_land 선행 실행 필요")
    o = pd.read_csv(po, encoding="utf-8-sig")[["date", "hour_n", "pred"]]
    h = pd.read_csv(ph, encoding="utf-8-sig")[["date", "hour_n", "pred"]]
    o["_src"], h["_src"] = "OOF", "홀드아웃"
    d = pd.concat([o, h], ignore_index=True).drop_duplicates(
        subset=["date", "hour_n"], keep="last")
    return d.rename(columns={"pred": "smp_land_pred"})


def add_stage2_features(df):
    """육지 연동 변수 생성. 전부 예측 시점 확보 가능한 정보만 사용."""
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    made = []

    # ① 전일 동시간 스프레드(제주 - 육지) — lag24
    df["spread_lag_d1"] = (df[TARGET] - df[LAND]).shift(24)
    made.append("spread_lag_d1")

    # ② 동시간 7일 평균 스프레드 — 최근 괴리 수준의 안정 추정
    df["spread_ma7"] = (df.groupby("hour_n", sort=False)["spread_lag_d1"]
                        .transform(lambda s: s.rolling(7, min_periods=3).mean()))
    made.append("spread_ma7")

    # ③ 전일 이탈 여부 — 이탈 상태의 지속 경향 반영
    df["is_decoupled_d1"] = (df["spread_lag_d1"].abs() > 1.0).astype("Int8")
    made.append("is_decoupled_d1")

    # ④ 최근 7일 이탈 빈도
    df["decoupled_rate7"] = (df.groupby("hour_n", sort=False)["is_decoupled_d1"]
                             .transform(lambda s: s.rolling(7, min_periods=3).mean()))
    made.append("decoupled_rate7")

    # ⑤ 제주 직접 추정치 = 육지 예측 + 최근 평균 스프레드
    df["jeju_est"] = df["smp_land_pred"] + df["spread_ma7"].fillna(0)
    made.append("jeju_est")

    # ⑥ 육지 예측의 전일 대비 변화율
    df["land_pred_ratio_d1"] = df["smp_land_pred"] / df[LAND].shift(24).replace(0, np.nan)
    made.append("land_pred_ratio_d1")

    made.append("smp_land_pred")
    return df, made


def build_folds(cv, folds, min_train_days):
    days = sorted(cv["date"].unique())
    n_min = min(min_train_days, max(180, len(days) - folds * 30))
    size = max(30, (len(days) - n_min) // folds)
    out = []
    for k in range(folds):
        tr = set(days[: n_min + size * k])
        va = set(days[n_min + size * k: n_min + size * (k + 1)])
        if va:
            out.append((cv[cv["date"].isin(tr)], cv[cv["date"].isin(va)], size))
    return out


def evaluate(name, feats, cv, ho, folds_def, kind, lgb, tuned, seed,
             residual=False):
    """교차검증 + 홀드아웃 평가.
    · residual=True : 타깃을 (제주 - 육지예측) 잔차로 학습하고
                      예측 시 육지예측을 더해 복원 — 육지 예측을 기준선으로 활용
    반환 : (교차검증 지표, 홀드아웃 지표, 홀드아웃 예측값, 모델)"""
    def _y(d):
        return d[TARGET] - d["smp_land_pred"] if residual else d[TARGET]

    def _restore(d, raw):
        return raw + d["smp_land_pred"] if residual else raw

    ms, oof = [], []
    for tr, va, _ in folds_def:
        m = make_model(kind, lgb, seed, tuned)
        m.fit(tr[feats], _y(tr))
        p = _restore(va, pd.Series(m.predict(va[feats]), index=va.index))
        ms.append(metrics(va[TARGET], p))
        oof.append(p)
    cvm = {k: float(np.nanmean([x[k] for x in ms])) for k in ms[0]}
    oof = pd.concat(oof) if oof else pd.Series(dtype=float)

    m = make_model(kind, lgb, seed, tuned)
    m.fit(cv[feats], _y(cv))
    p = _restore(ho, pd.Series(m.predict(ho[feats]), index=ho.index))
    hom = metrics(ho[TARGET], p)
    return cvm, hom, p, m, oof


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step4 제주 2단계 예측")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--min-train-days", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR)
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)

    SRC, DICT = feat_paths(TARGET)
    for p in (SRC, DICT):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    kind, lgb = load_model_class()
    tuned, src_p = load_tuned_params(kind, TARGET)

    print("=" * 78)
    print(" step4 제주 2단계 예측 — 육지 예측값 연동 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 엔진 : {kind} / 파라미터 : {src_p}")
    print("=" * 78)

    # ── ① 데이터 준비 ──────────────────────────
    print("\n■ ① 데이터 준비")
    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")
    base = [c for c in df.columns
            if c in set(dd[dd["운영가용"] == "Y"]["변수명"]) and c not in EXCLUDE]

    lp = load_land_pred()
    rep("육지 예측값", f"{len(lp):,}행",
        f"OOF {int((lp['_src'] == 'OOF').sum()):,} / 홀드아웃 {int((lp['_src'] == '홀드아웃').sum()):,}")
    df = df.merge(lp.drop(columns=["_src"]), on=["date", "hour_n"], how="left")

    df, made = add_stage2_features(df)
    rep("연동 변수", f"{len(made)}종", ", ".join(made))

    # 육지 예측값 보유 구간으로 한정
    df = df[df["smp_land_pred"].notna()].copy()
    df = df.dropna(subset=[TARGET])
    rep("유효 구간", f"{df['date'].min()} ~ {df['date'].max()}", f"{len(df):,}행")

    cats = [c for c in CATEGORICAL if c in base]
    for c in cats:
        df[c] = df[c].astype("category")
    base = [c for c in base if c in cats or is_num(df[c])]

    cv = df[df["date"] < HOLD_START].copy()
    ho = df[df["date"] >= HOLD_START].copy()
    rep("교차검증", f"{cv['date'].min()} ~ {cv['date'].max()}", f"{len(cv):,}행")
    rep("홀드아웃", f"{ho['date'].min()} ~ {ho['date'].max()}", f"{len(ho):,}행")

    folds_def = build_folds(cv, args.folds, args.min_train_days)
    rep("분할 구성", f"{len(folds_def)}개", f"검증창 {folds_def[0][2]}일")

    # ── ② A/B 비교 ─────────────────────────────
    print("\n■ ② A/B 비교 — 동일 구간·동일 조건")
    sets = {"A. 기존(육지 미연동)": (base, False),
            "B. 변수연동(육지예측 투입)": (base + made, False),
            "C. 잔차학습(육지예측 기준선)": (base + made, True)}
    res, preds, models, oofs = {}, {}, {}, {}
    for name, (feats, resid) in sets.items():
        cvm, hom, p, m, o = evaluate(name, feats, cv, ho, folds_def, kind, lgb,
                                     tuned, args.seed, residual=resid)
        res[name] = {"변수수": len(feats), "CV_MAPE": cvm["MAPE"],
                     **{f"HO_{k}": v for k, v in hom.items()}}
        preds[name], models[name], oofs[name] = p, m, o

    # D. 육지 예측값 직접 적용 — 모델 없는 기준선(신규 벤치마크)
    DN = "D. 육지예측 직접적용(기준선)"
    pd_direct = ho["smp_land_pred"]
    hod = metrics(ho[TARGET], pd_direct)
    cvd = metrics(cv.loc[oofs["C. 잔차학습(육지예측 기준선)"].index, TARGET],
                  cv.loc[oofs["C. 잔차학습(육지예측 기준선)"].index, "smp_land_pred"])
    res[DN] = {"변수수": 0, "CV_MAPE": cvd["MAPE"],
               **{f"HO_{k}": v for k, v in hod.items()}}
    preds[DN] = pd_direct

    # E. 앙상블 — 가중치는 교차검증 구간에서 결정(홀드아웃 미사용)
    CN = "C. 잔차학습(육지예측 기준선)"
    oc = oofs[CN]
    yv = cv.loc[oc.index, TARGET]
    od = cv.loc[oc.index, "smp_land_pred"]
    grid = [(w, metrics(yv, w * od + (1 - w) * oc)["MAPE"])
            for w in np.arange(0, 1.01, 0.05)]
    w_opt = float(min(grid, key=lambda t: t[1])[0])
    pe = w_opt * ho["smp_land_pred"] + (1 - w_opt) * preds[CN]
    hoe = metrics(ho[TARGET], pe)
    EN = f"E. 앙상블(w={w_opt:.2f}, CV결정)"
    res[EN] = {"변수수": len(sets[CN][0]), "CV_MAPE": min(g[1] for g in grid),
               **{f"HO_{k}": v for k, v in hoe.items()}}
    preds[EN] = pe

    print(f"  {'구분':<28}{'변수':>5}{'CV MAPE':>10}{'HO MAPE':>10}"
          f"{'RMSE':>9}{'MAE':>8}{'R²':>8}")
    print("  " + "-" * 78)
    for name, r in res.items():
        cvs = f"{r['CV_MAPE']:>10.2f}" if not np.isnan(r["CV_MAPE"]) else f"{'-':>10}"
        print(f"  {name:<28}{r['변수수']:>5}{cvs}{r['HO_MAPE']:>10.2f}"
              f"{r['HO_RMSE']:>9.2f}{r['HO_MAE']:>8.2f}{r['HO_R2']:>8.3f}")

    a = res["A. 기존(육지 미연동)"]
    # 최우수 선정 : MAPE 우선, 0.1%p 이내 동률군에서는 RMSE로 판정
    #  · 근거 : MAPE는 주지표이나 저가 구간 민감. 실질 동률이면 극단오차(RMSE) 억제 우선
    mmin = min(v["HO_MAPE"] for v in res.values())
    tie = {k: v for k, v in res.items() if v["HO_MAPE"] <= mmin + 0.1}
    bname, b = min(tie.items(), key=lambda kv: kv[1]["HO_RMSE"])
    if len(tie) > 1:
        print(f"  ※ MAPE 실질 동률 {len(tie)}개안 → RMSE 기준 판정")
    gain = a["HO_MAPE"] - b["HO_MAPE"]
    print("  " + "-" * 78)
    print(f"  최우수 : {bname}")
    print(f"  개선폭 : {gain:+.2f}%p  ({a['HO_MAPE']:.2f}% → {b['HO_MAPE']:.2f}%)")
    print(f"  Gate 3(MAPE ≤ 8.0%) : "
          f"{'달성' if b['HO_MAPE'] <= 8.0 else '미달'}")
    m = models.get(bname)
    feats_b = sets[bname][0] if bname in sets else []

    # ── ③ 변수 중요도 ──────────────────────────
    print(f"\n■ ③ 연동 변수 중요도 ({bname})")
    if m is not None and hasattr(m, "feature_importances_"):
        fi = pd.DataFrame({"변수": feats_b, "중요도": m.feature_importances_})
        fi["비중"] = fi["중요도"] / fi["중요도"].sum() * 100
        fi = fi.sort_values("중요도", ascending=False).reset_index(drop=True)
        fi["순위"] = fi.index + 1
        top = fi.head(10)
        for _, r in top.iterrows():
            mark = " ←연동" if r["변수"] in made else ""
            print(f"  {int(r['순위']):>3}. {r['변수']:<26}{r['비중']:>6.2f}%{mark}")
        w = fi[fi["변수"].isin(made)]
        print(f"  · 연동 변수 {len(made)}종 중요도 점유 : {w['비중'].sum():.1f}%")
        fi.to_csv(os.path.join(LOG_DIR,
                  f"step4_jeju_stage2_importance_{datetime.now():%Y%m%d_%H%M%S}.csv"),
                  index=False, encoding="utf-8-sig")

    # ── ④ 오차 분해 ────────────────────────────
    print(f"\n■ ④ 홀드아웃 오차 분해 (A 대비 {bname})")
    h = ho.copy()
    h["_pred"] = preds[bname]
    h["_predA"] = preds["A. 기존(육지 미연동)"]
    h["_이탈"] = np.where((h[TARGET] - h[LAND]).abs() <= 1.0, "동조(육지=제주)", "이탈")

    for key, label in (("_이탈", "육지 동조 여부"), ("daytype", "일유형")):
        if key not in h.columns:
            continue
        print(f"  [{label}]")
        for v, g in h.groupby(key, observed=True):
            mb = metrics(g[TARGET], g["_pred"])
            ma = metrics(g[TARGET], g["_predA"])
            print(f"    {str(v):<16} A {ma['MAPE']:>6.2f}%  →  B {mb['MAPE']:>6.2f}%"
                  f"  ({mb['MAPE'] - ma['MAPE']:+.2f}%p)  N {len(g):,}")

    print("  [월별]")
    for v, g in h.groupby(h["date"].str[:7], observed=True):
        mb = metrics(g[TARGET], g["_pred"])
        ma = metrics(g[TARGET], g["_predA"])
        print(f"    {v:<16} A {ma['MAPE']:>6.2f}%  →  B {mb['MAPE']:>6.2f}%"
              f"  ({mb['MAPE'] - ma['MAPE']:+.2f}%p)  N {len(g):,}")

    # ── ⑤ 저장 ─────────────────────────────────
    print("\n■ ⑤ 저장")
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    if gain <= 0:
        print("  [미저장] 개선 없음 — 기존 모델 유지")
    else:
        # E(앙상블)는 C 모델과 가중치 조합이므로 C 모델 + w 를 함께 보관
        if bname.startswith("E"):
            m, feats_b, resid, w = models[CN], sets[CN][0], sets[CN][1], w_opt
        elif bname.startswith("D"):
            m, feats_b, resid, w = None, [], False, 1.0
        else:
            m, feats_b, resid, w = models[bname], sets[bname][0], sets[bname][1], 0.0
        try:
            import joblib
            mp = os.path.join(MODEL_DIR, f"smp_{TARGET}_stage2.pkl")
            joblib.dump({"model": m, "features": feats_b, "categorical": cats,
                         "residual": resid, "land_weight": w, "method": bname,
                         "trained": ts, "ho_mape": b["HO_MAPE"]}, mp)
            print(f"  → 모델 저장 {mp}")
            print(f"     예측식 : 제주 = {w:.2f}×육지예측 + {1 - w:.2f}×"
                  + ("(육지예측 + 잔차모델)" if resid else "제주모델"))
        except ImportError:
            print("  [건너뜀] joblib 미설치")

    pd.DataFrame([{"구분": k, **v} for k, v in res.items()]).round(4).to_csv(
        os.path.join(LOG_DIR, f"step4_jeju_stage2_{ts}.csv"),
        index=False, encoding="utf-8-sig")
    # 전 방식 예측값 저장 — 사후 앙상블·구간별 진단에 활용
    sv = h[["date", "hour_n", TARGET, LAND, "smp_land_pred"]].copy()
    for i, (name, p) in enumerate(preds.items(), 1):
        sv[f"pred_{name.split('.')[0]}"] = p
    sv.to_csv(os.path.join(PROCESSED, "jeju_stage2_predictions.csv"),
              index=False, encoding="utf-8-sig")

    # 그래프
    if plt is not None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        d1 = h.groupby("date").agg(실적=(TARGET, "mean"), A=("_predA", "mean"),
                                   B=("_pred", "mean")).reset_index()
        x = pd.to_datetime(d1["date"])
        ax[0].plot(x, d1["실적"], lw=1.6, color="black", label="Actual")
        ax[0].plot(x, d1["A"], lw=1.2, color=C1, alpha=.8, label="A (base)")
        ax[0].plot(x, d1["B"], lw=1.2, color=C2, alpha=.8, label="B (stage2)")
        ax[0].set_title("Daily mean SMP - Jeju", fontsize=12, weight="bold")
        ax[0].legend(); ax[0].grid(alpha=.3)

        mm = h.groupby(h["date"].str[:7]).apply(
            lambda g: pd.Series({"A": metrics(g[TARGET], g["_predA"])["MAPE"],
                                 "B": metrics(g[TARGET], g["_pred"])["MAPE"]}),
            include_groups=False)
        idx = np.arange(len(mm))
        ax[1].bar(idx - .2, mm["A"], .4, color=C1, alpha=.85, label="A (base)")
        ax[1].bar(idx + .2, mm["B"], .4, color=C2, alpha=.85, label="B (stage2)")
        ax[1].axhline(8.0, color="green", ls="--", lw=1.2, label="Gate 3")
        ax[1].set_xticks(idx); ax[1].set_xticklabels(mm.index, rotation=45, fontsize=8)
        ax[1].set_title("Monthly MAPE (%)", fontsize=12, weight="bold")
        ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "01_2단계비교.png"), dpi=130)
        plt.close(fig)
        print(f"  → 그래프 {os.path.join(OUT_DIR, '01_2단계비교.png')}")

    print("=" * 78)
    print(f" 결론 : 제주 MAPE {a['HO_MAPE']:.2f}% → {b['HO_MAPE']:.2f}% ({gain:+.2f}%p)")
    print(f"        Gate 3 {'달성' if b['HO_MAPE'] <= 8.0 else '미달'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
