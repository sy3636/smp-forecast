# -*- coding: utf-8 -*-
"""
step4_train_ensemble.py — VST 앙상블 모델 학습 (v1.1)
==============================================================================
목적
  · 학술문헌 기반 4개 성분 모델을 학습하고 단순평균 앙상블로 저장
  · 과제 A-2(정확도목표 실행계획서 v1.2) 이행

구성 — 4개 성분 단순평균 + 운영단 편의보정
  | 성분 | 목적함수 | 타깃 변환 | 변수 | 근거 |
  |------|---------|----------|------|------|
  | ① vst    | L2 | asinh | 기본 | Uniejewski·Weron·Ziel(2018) |
  | ② vst_l1 | **L1(MAE)** | asinh | 기본 | **주지표 정렬 — v1.1 신설** |
  | ③ vst_x  | L2 | asinh | +교차시차 24종 | Raviv et al.(2013) |
  | ④ diff   | L2 | y − 전일동시간 | 기본 | 자체 실증(2026.9.2) |

v1.1 개정 (2026.9.3) — 아침 전환 구간 오차 대응
  · 배경 : 08~10시 급전 전환(석탄→LNG) 시각이 날마다 이동하여 계단 60~70원이
          통째로 어긋나는 **위상(phase) 오차** 발생
          예) 2026.9.1~9.2는 9시 급등(+62~65원), 9.3은 10시 급등(+69원)
          전환 시각 표준편차 **1.35시간**(2026.6~9 실측, 94일)
  · 조치 : 성분 ②를 **최근성 가중(vst_w) → L1 목적함수(vst_l1)** 로 교체
          L2는 조건부 평균, L1은 조건부 중앙값을 추정하므로
          전환 시각이 양분되는 구간에서 **MAE 기준 우월**
  · 실증 (홀드아웃 2026-01-29~09-02, 편의보정 포함)
    | 구성 | 전체 MAE | 전체 rMAE | 08~10시 MAE | 08~10시 rMAE |
    |------|---------|----------|------------|-------------|
    | v1.0 (vst_w) | 8.06 | 0.744 | 9.53 | 0.632 |
    | **v1.1 (vst_l1)** | **7.97** | **0.736** | **9.41** | **0.624** |
  · 기각 : 전환위상 변수 6종(ramp_hour·hours_from_ramp 등)
          08~10시 MAE 9.63 → 9.76 악화. 전환 시각 자체의 예측 가능성이 낮아 잡음으로 작용

분산안정화변환(VST)
    z = arcsinh( (y − median) / (1.4826 × MAD) )
    ŷ = sinh(ẑ) × (1.4826 × MAD) + median
  · 계수(median·MAD)는 **학습 구간에서만** 산출하여 누출 차단
  · 모델 파일에 계수를 동봉 저장하여 예측 시 동일 역변환 보장

실증 결과 (2026.9.2, 홀드아웃 2026-01-29~09-02 N=5,184)
  | 구성 | MAE | rMAE | MAPE | 편의 |
  |------|-----|------|------|------|
  | 현행 단일 모델            | 8.72 | 0.804 | 8.24% | +1.44% |
  | 현행 + 편의보정           | 8.45 | 0.779 | 8.01% | +1.15% |
  | **본 앙상블**             | 8.36 | 0.771 | 7.99% | +1.95% |
  | **본 앙상블 + 편의보정**   | **8.04** | **0.742** | **7.66%** | **+1.38%** |
  · 급등구간(2026-08~) : MAE 8.70 / rMAE 0.865 / MAPE 6.44%

기각 기법 (동일 실증)
  · VST log        : MAE 9.50 — 저가 구간 역변환 시 오차 증폭
  · 시간대별 24모델 : MAE 9.70 — 표본 분할에 따른 학습 손실
  · 전주 교차시차 추가 : MAE 8.63 — 전일 단독 대비 열위

적용 범위
  · **시간별 예측(D+1, 23:30)** 전용
  · 주간 예측(15시, D+2 이상)은 전일 실적 미확보로 교차시차 사용 불가 → 현행 유지

실행
  python scripts\\step4_train_ensemble.py
  python scripts\\step4_train_ensemble.py --target smp_jeju
  python scripts\\step4_train_ensemble.py --no-cross      (교차시차 성분 제외)

산출
  · ml/models/smp_{타깃}_ensemble.pkl
  · logs/step4_train_ensemble_{타깃}_{일시}.csv

v1.2 개정 (2026.9.3) — 주간 D+1 축약 앙상블 실증 지원
  · 배경 : 주간 예측(15:00)은 수요예측 미확보·전일 실적 미확보로
          시간별 앙상블(성분 4종)을 그대로 적용 불가
  · 조치 : --no-demand(수요예측 12종 제외) · --tag(산출물 분리) · --no-save 신설
  · 주간 D+1 실증 : --no-cross --no-demand 조합 → 성분 3종(vst·vst_l1·diff) 축약 구성
  · 운영 모델 보호 : --tag 미지정 시에만 운영 경로(smp_{타깃}_ensemble.pkl) 사용

버전 : v1.2 (2026-09-03)
실행(주간 D+1 실증)
  python scripts\\step4_train_ensemble.py --no-cross --no-demand --tag _weekly --no-save
"""
import os
import sys
import argparse
import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_baseline import (  # noqa: E402
    feat_paths, metrics, TRAIN_START, CV_END, HOLD_START, set_mape_floor,
)
from step3_cross_hour import add_cross_hour  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

CATEGORICAL = ["daytype", "season"]
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju",
           "smp_land_raw", "smp_jeju_raw"}
LGB = dict(n_estimators=1200, learning_rate=0.05, num_leaves=63,
           min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
           reg_lambda=1.0, random_state=42)
HALF_LIFE = 730          # 최근성 가중 반감기(일) — v1.1에서 미사용(참고 보존)
DIFF_BASE = "smp_lag_d1"  # 차분 타깃 기준선

_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {값:<24} {비고}")
    _rows.append({"항목": 항목, "값": 값, "비고": 비고})


def vst_coef(y):
    """VST 계수 산출 — 중앙값·중앙절대편차(정규 등가 척도)"""
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med))) * 1.4826
    return med, (mad if mad > 1e-6 else 1.0)


def vst(y, med, mad):
    return np.arcsinh((np.asarray(y, float) - med) / mad)


def ivst(z, med, mad):
    return np.sinh(np.asarray(z, float)) * mad + med


def make_model(objective="l2"):
    """성분 모델 생성 — objective 'l1' 은 MAE 목적함수(주지표 정렬)"""
    try:
        from lightgbm import LGBMRegressor
        kw = dict(LGB)
        if objective == "l1":
            kw["objective"] = "mae"
        return LGBMRegressor(verbose=-1, **kw), f"lightgbm/{objective}"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=30, l2_regularization=1.0, random_state=42), "sklearn"


def evaluate(y, p, naive):
    """4지표 산출 — PRD v2.13"""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n = np.asarray(naive, float)
    m = np.isfinite(n)
    den = np.maximum(y, 20.0)
    mae = float(np.abs(p - y).mean())
    mae0 = float(np.abs(n[m] - y[m]).mean()) if m.any() else np.nan
    return {
        "MAE": round(mae, 2),
        "rMAE": round(mae / mae0, 3) if mae0 and np.isfinite(mae0) else np.nan,
        "MAPE": round(float((np.abs(p - y) / den).mean() * 100), 2),
        "편의": round(float(((p - y) / den).mean() * 100), 2),
    }


def main():
    ap = argparse.ArgumentParser(description="VST 앙상블 학습")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--no-cross", action="store_true",
                    help="교차시차 성분(③) 제외")
    ap.add_argument("--no-demand", action="store_true",
                    help="수요예측 계열(demand_fcst*) 제외 — 주간 예측(15시) 조건")
    ap.add_argument("--tag", default="",
                    help="산출물 접미사(예: _weekly) — 운영 모델과 분리 저장")
    ap.add_argument("--variant", default="",
                    help="입력 피처셋 접미사(예: _lagwfill) — 피처 A/B 실증용. "
                         "미지정 시 운영 피처셋 사용")
    ap.add_argument("--no-save", action="store_true",
                    help="모델 미저장 — 성능 비교(A/B) 전용 실행")
    ap.add_argument("--mape-floor", type=float, default=20.0)
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    print("=" * 78)
    print(" step4_train_ensemble — VST 앙상블 학습 v1.1")
    print(f" 실행시각 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    SRC, DICT = feat_paths(t)
    #  --variant 지정 시 실증용 피처셋으로 대체
    #   · 2026.9.6 결함 : --tag 는 산출물 접미사일 뿐 입력 피처셋을 바꾸지 않아
    #     피처 A/B 실증이 동일 파일 2회 학습으로 수행되어 무효였음
    if args.variant:
        v_src = os.path.join(PROCESSED, f"smp_features_{t}{args.variant}.csv")
        v_dic = os.path.join(PROCESSED, f"feature_dictionary_{t}{args.variant}.csv")
        if not os.path.exists(v_src):
            sys.exit(f"[중단] 실증 피처셋 없음 → {v_src}")
        SRC = v_src
        DICT = v_dic if os.path.exists(v_dic) else DICT
    for p in (SRC, DICT):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")
    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")

    print("\n■ ① 변수 선정 (운영가용 = Y)")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    base = [c for c in df.columns if c in ok and c not in EXCLUDE]
    cats = [c for c in CATEGORICAL if c in base]
    base = [c for c in base
            if c in cats or pd.api.types.is_numeric_dtype(df[c])]
    for c in cats:
        df[c] = df[c].astype("category")
    #  수요예측 의존 계열 — 주간예측(15시)에서는 순부하 파생도 동시 제외
    #  · A-5(2026.9.8) : net_load_* 는 demand_fcst_land 에서 파생 → 동일 제약
    demand = [c for c in base
              if c.startswith("demand_fcst") or c.startswith("net_load")
              or c == "pv_penetration"]
    if args.no_demand:
        base = [c for c in base if c not in demand]
    rep("기본 변수", f"{len(base)}개", f"범주 {len(cats)}개")
    rep("수요예측 의존 계열", f"{len(demand)}개",
        "제외 — 예측 시각(15시)에 미확보" if args.no_demand
        else "사용 — 수요예측·순부하 포함")
    rep("시나리오", "주간 D+1(15시)" if args.no_demand else "시간별 D+1(23:30)",
        "축약 앙상블 실증" if args.no_demand else "운영 구성")

    df, cross = add_cross_hour(df, t)
    use_cross = (not args.no_cross) and len(cross) == 24
    rep("교차시차 변수", f"{len(cross)}개",
        "성분 ③ 사용" if use_cross else "미사용")

    tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)]
    tr = tr.dropna(subset=[t]).copy()
    ho = df[df["date"] >= HOLD_START].dropna(subset=[t]).copy()
    rep("학습 구간", f"{tr['date'].min()} ~ {tr['date'].max()}", f"{len(tr):,}행")
    rep("홀드아웃", f"{ho['date'].min()} ~ {ho['date'].max()}", f"{len(ho):,}행")

    # ── ② 성분 학습 ────────────────────────────
    print("\n■ ② 성분 학습")
    med, mad = vst_coef(tr[t].values)
    rep("VST 계수", f"median {med:.2f} / MAD {mad:.2f}", "학습 구간 산출 — 누출 없음")

    age = (pd.to_datetime(CV_END) - pd.to_datetime(tr["date"])).dt.days.values
    w = 0.5 ** (age / HALF_LIFE)

    members = []
    #  성분 정의 — (이름, 타깃변환, 표본가중, 변수, 목적함수)
    #  v1.1 : vst_w(최근성 가중) → vst_l1(L1 목적함수) 교체
    specs = [("vst", "asinh", None, base, "l2"),
             ("vst_l1", "asinh", None, base, "l1"),
             ("diff", "diff", None, base, "l2")]
    if use_cross:
        specs.insert(2, ("vst_x", "asinh", None, base + cross, "l2"))

    for name, kind, sw, feats, obj in specs:
        mdl, engine = make_model(obj)
        if kind == "asinh":
            ytr = vst(tr[t].values, med, mad)
            mdl.fit(tr[feats], ytr, sample_weight=sw)
        else:
            m_ok = tr[DIFF_BASE].notna()
            mdl.fit(tr.loc[m_ok, feats],
                    tr.loc[m_ok, t].values - tr.loc[m_ok, DIFF_BASE].values,
                    sample_weight=None if sw is None else sw[m_ok.values])
        members.append({"name": name, "kind": kind, "features": feats,
                        "model": mdl, "categorical": cats})
        rep(f"  성분 {name}", f"{len(feats)}변수", f"{kind} / {engine}")

    # ── ③ 홀드아웃 평가 ────────────────────────
    print("\n■ ③ 홀드아웃 평가")

    def predict_all(d):
        ps = []
        for mb in members:
            if mb["kind"] == "asinh":
                ps.append(ivst(mb["model"].predict(d[mb["features"]]), med, mad))
            else:
                b = d[DIFF_BASE].fillna(float(np.nanmean(d[t]))).values
                ps.append(mb["model"].predict(d[mb["features"]]) + b)
        return np.mean(ps, axis=0), ps

    ens, parts = predict_all(ho)
    nv = ho[DIFF_BASE].values
    print(f"  {'구성':<14}{'MAE':>8}{'rMAE':>8}{'MAPE':>9}{'편의':>9}")
    print("  " + "-" * 48)
    for mb, p in zip(members, parts):
        m = evaluate(ho[t], p, nv)
        print(f"  {mb['name']:<14}{m['MAE']:>8.2f}{m['rMAE']:>8.3f}"
              f"{m['MAPE']:>8.2f}%{m['편의']:>8.2f}%")
        rep(f"성분 {mb['name']}", f"MAE {m['MAE']} / rMAE {m['rMAE']}",
            f"MAPE {m['MAPE']}% / 편의 {m['편의']}%")
    me = evaluate(ho[t], ens, nv)
    print("  " + "-" * 48)
    print(f"  {'앙상블':<14}{me['MAE']:>8.2f}{me['rMAE']:>8.3f}"
          f"{me['MAPE']:>8.2f}%{me['편의']:>8.2f}%")
    rep("앙상블", f"MAE {me['MAE']} / rMAE {me['rMAE']}",
        f"MAPE {me['MAPE']}% / 편의 {me['편의']}%")

    # 단계 목표 대비
    #  목표 기준 — 산출 조건별 상이(목표오차 확정서 v1.1 반영 예정)
    #    · 시간별 D+1(23:30, 수요예측 사용) : MAE ≤ 8.0 · rMAE ≤ 0.75
    #    · 주간   D+1(15:00, 수요예측 미사용) : MAE ≤ 8.5 · rMAE 목표 미설정
    tgt_mae = 8.5 if args.no_demand else 8.0
    scen_nm = "주간 D+1" if args.no_demand else "시간별 D+1"
    print(f"\n■ ④ 1단계 목표 대비 ({scen_nm} · MAE ≤ {tgt_mae:.1f})")
    ok_mae = me["MAE"] <= tgt_mae
    ok_rmae = me["rMAE"] <= 0.75
    print(f"  · MAE  {me['MAE']:.2f} / {tgt_mae:.2f}   {'○' if ok_mae else '×'}")
    print(f"  · rMAE {me['rMAE']:.3f} / 0.750  {'○' if ok_rmae else '×'}"
          + ("   (참고 — 주간은 rMAE 목표 미설정)" if args.no_demand else ""))
    print("  ※ 운영단 편의보정 적용 시 추가 개선 예상(실증 rMAE −0.029)")
    rep("1단계 판정(보정 전)", "달성" if (ok_mae and ok_rmae) else "미달",
        "편의보정 적용 후 재판정")

    # ── ⑤ 저장 ─────────────────────────────────
    print("\n■ ⑤ 저장")
    os.makedirs(MODEL_DIR, exist_ok=True)
    pack = {
        "members": members,
        "vst": {"median": med, "mad": mad},
        "diff_base": DIFF_BASE,
        "features": sorted({c for mb in members for c in mb["features"]}),
        "categorical": cats,
        "weights": [1.0 / len(members)] * len(members),
        "target": t,
        "trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "holdout": me,
        "scenario": "주간 D+1(15시)" if args.no_demand else "시간별 D+1(23:30)",
        "note": ("VST 앙상블 v1.2 — 주간 D+1 축약 구성(수요예측·교차시차 제외)"
                 if args.no_demand else
                 "VST 앙상블 v1.2 — 시간별 예측(D+1) 전용, L1 성분 포함"),
    }
    if args.no_save:
        rep("모델 파일", "미저장", "성능 비교(A/B) 전용 실행")
    else:
        op = os.path.join(MODEL_DIR, f"smp_{t}_ensemble{args.tag}.pkl")
        joblib.dump(pack, op)
        rep("모델 파일", os.path.relpath(op, BASE_DIR),
            f"성분 {len(members)}종 / 변수 {len(pack['features'])}종")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step4_train_ensemble_{t}_{ts}.csv")
    pd.DataFrame(_rows).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")


if __name__ == "__main__":
    main()
