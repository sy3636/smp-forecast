# -*- coding: utf-8 -*-
"""
step8_wx_daytime_test.py — 기상·신재생 변수 주간 한정 반영 실증 (v1.0)
==============================================================================
목적
  · 기상·신재생 정보를 **주간 시간대에 한정**하여 반영할 경우의 SMP 예측 개선 여부 판정
  · 2026.8.31 부서장 제안("신재생 영향은 주간에만 유효하므로 주간에만 반영") 검증

배경
  · 종전 실증(2026.8.6, 8.31)에서 기상·신재생 변수의 전 시간대 일괄 투입은 기각
  · 잔차 진단 결과, 오차의 **방향**과는 무상관(|r|<0.07)이나 **크기**와는 상관 존재
    (solar_land +0.234, pv_fcst +0.214) → 시간대 한정 반영의 타당성 검토 필요

설계 — 대조군 병행으로 개선 원인 분리
  · 단일모델      : 현행 구조(기상 미투입)
  · 2분할         : 주간(9~17시)·야간 분리 학습, 기상 미투입 **대조군**
  · 2분할+PV      : 2분할 후 **주간 모델에만** 태양광 전망(pv_fcst) 투입
  · 대조군을 두지 않으면 '분할 효과'와 '기상 효과'가 혼재되어 오판 위험

판정 기준
  · G-2 준용 — 전체 MAPE 0.10%p 이상 개선 시 채택
  · 시드 3종 × 홀드아웃 2구간에서 일관 재현되어야 함(시드 편차 초과분만 인정)

산출
  · data/processed/wx_daytime_test.csv — 조건별 성능
  · logs/step8_wx_daytime_test_{일시}.csv

실행
  python scripts\\step8_wx_daytime_test.py
  python scripts\\step8_wx_daytime_test.py --seeds 42,7,2026

버전 : v1.0 (2026-08-31)
"""
import os
import sys
import gc
import argparse
import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("[중단] lightgbm 미설치 — pip install lightgbm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

SRC = os.path.join(PROCESSED, "smp_features_smp_land_anom.csv")
DICT = os.path.join(PROCESSED, "feature_dictionary_smp_land.csv")
OUT = os.path.join(PROCESSED, "wx_daytime_test.csv")

TARGET = "smp_land"
WX = ["pv_fcst"]                 # 주간 한정 투입 대상 — 태양광 전망
DAY_LO, DAY_HI = 9, 17           # 주간 정의 — 태양광 출력 집중 구간

# 홀드아웃 구간 — (명칭, 학습종료일, 검증종료일)
HOLDOUTS = [("2026-01~08", "2026-01-01", None),
            ("2025-09~12", "2025-09-01", "2026-01-01")]

_rows_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {값:<26} {비고}", flush=True)
    _rows_log.append({"항목": 항목, "값": 값, "비고": 비고})


def load():
    """데이터 적재 — 필요 컬럼만 선별하여 메모리 절감

    비고 : 전체 116열 적재 시 메모리 초과로 프로세스 종료 발생(2026.8.31)
    """
    dic = pd.read_csv(DICT, encoding="utf-8-sig")
    key = dic.columns[0]
    feats = [c for c in dic[dic["운영가용"] == "Y"][key] if c != TARGET]

    head = pd.read_csv(SRC, encoding="utf-8-sig", nrows=1)
    use = [c for c in set(feats + WX + [TARGET, "date", "hour_n"]) if c in head.columns]
    d = pd.read_csv(SRC, encoding="utf-8-sig", usecols=use, low_memory=False)

    d = d[d[TARGET] > 1].reset_index(drop=True)     # SMP 0 구간 제외(MAPE 정의 불가)
    dt = d["date"].astype(str).copy()               # 문자열 원본 보존 — 범주형 변환 전 분할용
    feats = [c for c in feats if c in d.columns]

    # 메모리 절감 — 실수형 float32 축소, 문자형 범주화
    for c in feats + WX:
        if c not in d.columns:
            continue
        if d[c].dtype == object:
            d[c] = d[c].astype("category")
        elif d[c].dtype == "float64":
            d[c] = d[c].astype("float32")
    return d, dt, feats


def fit_predict(tr, te, feats, target, seed):
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                          random_state=seed, verbose=-1, n_jobs=2, force_col_wise=True)
    m.fit(tr[feats], tr[target])
    p = m.predict(te[feats])
    del m
    gc.collect()
    return p


def mape(pred, act):
    return float((np.abs(pred - act) / act).mean() * 100)


def run_one(d, day_mask, m_tr, m_te, feats, seed):
    """단일 (홀드아웃 × 시드) 조합 — 3개 구조 성능 산출"""
    tr, te = d[m_tr], d[m_te]
    d_tr, d_te = day_mask[m_tr], day_mask[m_te]
    act = te[TARGET].values
    out = {}

    # ① 단일모델 — 현행
    out["단일모델"] = mape(fit_predict(tr, te, feats, TARGET, seed), act)

    # ②③ 시간대 분할 — 기상 미투입(대조군) / 주간에만 투입
    for tag, use_wx in [("2분할", False), ("2분할+PV", True)]:
        p = np.zeros(len(te))
        for is_day in (True, False):
            f = feats + (WX if (use_wx and is_day) else [])
            p[d_te == is_day] = fit_predict(tr[d_tr == is_day], te[d_te == is_day],
                                            f, TARGET, seed)
        out[tag] = mape(p, act)
    return out


def save_log(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(LOG_DIR, f"{name}_{ts}.csv")
    pd.DataFrame(_rows_log).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def main():
    ap = argparse.ArgumentParser(description="기상·신재생 주간 한정 반영 실증")
    ap.add_argument("--seeds", default="42,7,2026", help="난수 시드(쉼표 구분)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 74)
    print(" step8_wx_daytime_test — 기상·신재생 주간 한정 반영 실증 v1.0")
    print("=" * 74)

    for f in (SRC, DICT):
        if not os.path.exists(f):
            sys.exit(f"[중단] 입력파일 없음 → {f}")

    print("\n■ ① 데이터 적재")
    d, dt, feats = load()
    rep("데이터", f"{len(d):,}행", f"{dt.min()} ~ {dt.max()}")
    rep("운영가용 변수", f"{len(feats)}종", f"주간 추가 투입 : {', '.join(WX)}")
    rep("주간 정의", f"{DAY_LO}~{DAY_HI}시", "태양광 출력 집중 구간")

    day_mask = d["hour_n"].between(DAY_LO, DAY_HI).values

    print("\n■ ② 조건별 실증 — 시드 × 홀드아웃")
    print(f"\n  {'홀드아웃':<12}{'시드':>7}{'단일모델':>11}{'2분할':>11}{'2분할+PV':>12}")
    rows = []
    for lab, tr_end, te_end in HOLDOUTS:
        m_tr = (dt < tr_end).values
        m_te = (dt >= tr_end).values if te_end is None else ((dt >= tr_end) & (dt < te_end)).values
        for s in seeds:
            r = run_one(d, day_mask, m_tr, m_te, feats, s)
            print(f"  {lab:<12}{s:>7}{r['단일모델']:>11.2f}{r['2분할']:>11.2f}{r['2분할+PV']:>12.2f}",
                  flush=True)
            rows.append({"홀드아웃": lab, "시드": s, **{k: round(v, 3) for k, v in r.items()}})

    e = pd.DataFrame(rows)

    print("\n■ ③ 판정")
    for lab, _, _ in HOLDOUTS:
        g = e[e["홀드아웃"] == lab]
        a, b, c = g["단일모델"].mean(), g["2분할"].mean(), g["2분할+PV"].mean()
        sd = g["단일모델"].std()
        rep(f"[{lab}] 단일모델", f"{a:.2f}%", f"시드 편차 ±{sd:.2f}%p")
        rep(f"[{lab}] 2분할", f"{b:.2f}% ({b - a:+.2f}%p)",
            "채택" if (a - b) >= 0.10 else "기준 미달")
        rep(f"[{lab}] 2분할+PV", f"{c:.2f}% ({c - a:+.2f}%p)",
            "채택" if (a - c) >= 0.10 else "기준 미달")
        rep(f"[{lab}] PV 순기여", f"{c - b:+.2f}%p", "2분할 대비 — 기상 단독 효과")

    print("\n■ ④ 산출")
    os.makedirs(PROCESSED, exist_ok=True)
    e.to_csv(OUT, index=False, encoding="utf-8-sig")
    rep("성능 파일", os.path.relpath(OUT, BASE_DIR), f"{len(e)}개 조합")

    p = save_log("step8_wx_daytime_test")
    print(f"\n[완료] 로그 → {os.path.relpath(p, BASE_DIR)}")


if __name__ == "__main__":
    main()
