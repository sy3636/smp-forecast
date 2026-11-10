# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step6 변수 중요도 산출
================================================
목적 : 학습 모형의 변수 기여도(Gain)를 산출하여 변수군 채택·기각 판단 근거를 제공
       (2026.8.27 시연 지적 대응 — 연료비 구조 변수 실증의 설명자료)

설계
  · step4_train_horizon 과 동일한 변수 선정·모델 구성을 재사용하여 조건 일치 확보
  · Gain 기준 중요도 — 분기 시 감소한 손실 총합(변수의 실제 기여)
  · 변수군(prefix) 단위 집계 병행 — 개별 변수 순위와 군 단위 기여 동시 확인

입력 : data/processed/smp_features_{타깃}{variant}.csv
       data/processed/feature_dictionary_{타깃}{variant}.csv
출력 : logs/step6_importance_{타깃}{variant}_{일시}.csv

버전 : v1.0 (2026-08-29)
실행 : python scripts\\step6_importance.py --target smp_land --variant _fuel
       python scripts\\step6_importance.py --target smp_land --variant _fuel --train-start 2025-01-01
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_train_horizon import (CATEGORICAL, EXCLUDE, CV_END, TRAIN_START,  # noqa: E402
                                 is_num, load_tuned, make_model)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 변수군 분류 — 접두어 기준
GROUPS = [("fc_", "연료비 구조"), ("smp_", "SMP 시차"), ("demand_", "수요"),
          ("temp_", "기상"), ("grp_", "가격결정발전기"), ("is_", "달력 지시"),
          ("hol", "휴일"), ("hour", "시간"), ("month", "월"), ("dow", "요일")]


def group_of(name):
    for p, g in GROUPS:
        if name.startswith(p):
            return g
    return "기타(달력·순환)"


def main():
    ap = argparse.ArgumentParser(description="변수 중요도 산출")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--variant", default="", help="피처셋 접미사(예: _fuel)")
    ap.add_argument("--horizon", type=int, default=1, help="예측 지평(일)")
    ap.add_argument("--train-start", default=None, help="학습 시작일 재지정(YYYY-MM-DD)")
    ap.add_argument("--top", type=int, default=25, help="출력 상위 변수 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t, v = args.target, args.variant

    src = os.path.join(PROCESSED, f"smp_features_{t}{v}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}{v}.csv")
    for p in (src, dic):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")

    print("=" * 78)
    print(" step6 변수 중요도 산출 (v1.0)")
    print(f" 타깃 : {t}{v or '(운영 기준)'} / 지평 : D+{args.horizon}")
    print("=" * 78)

    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(dic, encoding="utf-8-sig")

    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in ok and c not in EXCLUDE and is_num(df[c])]
    feats = [c for c in feats if not c.startswith("demand_fcst")]  # 예측 시각 미확보
    for c in [c for c in CATEGORICAL if c in feats]:
        df[c] = df[c].astype("category")

    shift = 24 * (args.horizon - 1)
    df["_y"] = df[t].shift(-shift)
    ts = args.train_start or TRAIN_START
    tr = df[(df["date"] >= ts) & (df["date"] <= CV_END)].dropna(subset=["_y"])
    print(f"\n■ ① 학습 : {len(tr):,}행 ({ts} ~ {CV_END}) / 변수 {len(feats)}개")

    params, psrc = load_tuned(t)
    m, kind = make_model(params, args.seed)
    m.fit(tr[feats], tr["_y"])

    if kind != "lightgbm":
        sys.exit("[중단] Gain 중요도는 LightGBM 필요 → pip install lightgbm")
    gain = m.booster_.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"변수명": feats, "gain": gain})
    imp["비중"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
    imp["변수군"] = imp["변수명"].map(group_of)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["순위"] = imp.index + 1
    imp = imp.merge(dd[["변수명", "정의"]].drop_duplicates("변수명"), on="변수명", how="left")

    print(f"\n■ ② 상위 {args.top}개 변수")
    print(f"  {'순위':<5}{'변수명':<26}{'비중':>7}  정의")
    print("  " + "-" * 74)
    for _, r in imp.head(args.top).iterrows():
        print(f"  {r['순위']:<5}{r['변수명']:<26}{r['비중']:>6.2f}%  {str(r['정의'])[:34]}")

    print("\n■ ③ 변수군별 기여도")
    g = imp.groupby("변수군").agg(변수수=("변수명", "size"), 비중=("비중", "sum")).round(2)
    g = g.sort_values("비중", ascending=False)
    for k, r in g.iterrows():
        print(f"  · {k:<16}{int(r['변수수']):>4}종{r['비중']:>8.2f}%")

    fc = imp[imp["변수군"] == "연료비 구조"]
    if len(fc):
        print(f"\n■ ④ 연료비 구조 변수 요약")
        print(f"  · 총 기여 비중 : {fc['비중'].sum():.2f}%  ({len(fc)}종)")
        print(f"  · 최상위 변수  : {fc.iloc[0]['변수명']} (전체 {int(fc.iloc[0]['순위'])}위, "
              f"{fc.iloc[0]['비중']:.2f}%)")
        print(f"  · 상위 20위 이내 : {int((fc['순위'] <= 20).sum())}종")

    os.makedirs(LOG_DIR, exist_ok=True)
    out = os.path.join(LOG_DIR,
                       f"step6_importance_{t}{v}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    imp[["순위", "변수명", "변수군", "gain", "비중", "정의"]].to_csv(
        out, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 결과 → {os.path.relpath(out, BASE_DIR)}")


if __name__ == "__main__":
    main()
