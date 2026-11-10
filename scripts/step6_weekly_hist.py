# -*- coding: utf-8 -*-
"""
step6_weekly_hist.py — 주간 예측 이력 누적·실적 재결합 (v1.0)
==============================================================================
목적
  · 주간 예측(15:00, D+1~D+6) 산출분을 단일 이력 파일로 누적하고,
    산출 시점에 미확보인 실적을 사후 재결합하여 **지평별 잔차 이력**을 확보

배경 (2026.9.3 주간 편의보정 확대 설계서 v1.0)
  · 주간 산출물은 `SMP주간예측_{기준일}.csv` 일자별 파일로만 저장되고,
    실적은 산출 시점 미확보 상태로 기록되어 잔차 산정이 불가
  · 시간별 예측은 `predictions_{타깃}.csv` 누적 파일을 보유하여 편의보정이 가능하나,
    주간은 이력 부재로 **편의보정 확대의 선결 조건 미충족**
  · 지평별 잔차 이력 확보 시 주간 D+2~D+6 편의보정 적용 가능
    (예비 백테스트 : K=14·γ=0.7에서 MAE 12.11 → 11.81, −2.4%)

설계
  · 일자별 파일 전량을 스캔하여 재구성 — append 누락·중복 위험 없음
  · 실적은 `smp_dataset_hourly.csv`에서 (대상일, 거래시간) 기준으로 재결합
  · 동일 (기준일·지평·대상일·거래시간) 중복 시 최신 파일분 채택
  · 산출 : outputs/predict/weekly_predictions_{타깃}.csv

컬럼
  기준일 · 지평 · 대상일 · 거래시간 · 예측 · 실적 · 잔차(실적−예측) · 갱신시각

실행
  python scripts\\step6_weekly_hist.py                    (육지 — 기본)
  python scripts\\step6_weekly_hist.py --target smp_jeju
  python scripts\\step6_weekly_hist.py --summary          (지평별 현황 출력)

비고 : step6_weekly_predict.py 산출 직후 자동 호출됨

버전 : v1.0 (2026-09-05)
"""
import os
import sys
import glob
import argparse
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATASET = os.path.join(PROCESSED, "smp_dataset_hourly.csv")

COLS = ["기준일", "지평", "대상일", "거래시간", "예측", "실적", "잔차", "갱신시각"]


def hist_path(t):
    return os.path.join(OUT_DIR, f"weekly_predictions_{t}.csv")


def load_actual(t):
    """실적 적재 — 대상일·거래시간 기준"""
    if not os.path.exists(DATASET):
        return pd.DataFrame(columns=["대상일", "거래시간", "실적"])
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c in ("date", "hour_n", t))
    if t not in d.columns:
        return pd.DataFrame(columns=["대상일", "거래시간", "실적"])
    d = d.rename(columns={"date": "대상일", "hour_n": "거래시간", t: "실적"})
    d["대상일"] = d["대상일"].astype(str)
    return d.dropna(subset=["실적"])


def rebuild(t, verbose=True):
    """일자별 산출물 전량을 재구성하고 실적을 재결합"""
    files = sorted(glob.glob(os.path.join(OUT_DIR, "SMP주간예측_*.csv")))
    if not files:
        if verbose:
            print("  · 주간 산출물 없음 — 이력 생성 생략")
        return None

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        need = {"기준일", "지평", "대상일", "거래시간", "예측"}
        if not need.issubset(d.columns):
            continue
        frames.append(d[["기준일", "지평", "대상일", "거래시간", "예측"]])
    if not frames:
        if verbose:
            print("  · 형식 일치 산출물 없음 — 이력 생성 생략")
        return None

    h = pd.concat(frames, ignore_index=True)
    h["대상일"] = h["대상일"].astype(str)
    h["기준일"] = h["기준일"].astype(str)
    h["거래시간"] = pd.to_numeric(h["거래시간"], errors="coerce").astype("Int64")

    #  동일 (기준일·지평·대상일·거래시간) 중복 시 최신 파일분 채택
    h = h.drop_duplicates(subset=["기준일", "지평", "대상일", "거래시간"],
                          keep="last")

    #  실적 재결합 — 산출 시점 미확보분을 사후 반영
    act = load_actual(t)
    h = h.merge(act, on=["대상일", "거래시간"], how="left")
    h["잔차"] = h["실적"] - h["예측"]
    h["갱신시각"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    h = h.sort_values(["대상일", "거래시간", "지평"]).reset_index(drop=True)

    p = hist_path(t)
    tmp = p + ".tmp"
    h[COLS].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, p)                      # 원자적 저장 — 중단 시 손상 방지

    if verbose:
        n_act = int(h["실적"].notna().sum())
        print(f"  · 이력 누적 : {len(h):,}행 / 산출물 {len(files)}건")
        print(f"  · 실적 결합 : {n_act:,}행 ({n_act / len(h) * 100:.1f}%)")
        print(f"  → 저장 {p}")
    return h


def summary(t):
    """지평별 이력 현황 — 편의보정 적용 요건 점검"""
    p = hist_path(t)
    if not os.path.exists(p):
        print(f"[중단] 이력 없음 → {p}")
        return
    h = pd.read_csv(p, encoding="utf-8-sig")
    v = h.dropna(subset=["실적"])
    print(f"\n■ 지평별 이력 현황 ({t})")
    print(f"  {'지평':<6}{'관측일수':>9}{'행수':>8}{'MAE':>9}{'평균잔차':>10}")
    print("  " + "-" * 44)
    for g in sorted(v["지평"].dropna().unique()):
        s = v[v["지평"] == g]
        print(f"  {g:<6}{s['대상일'].nunique():>9}{len(s):>8}"
              f"{s['잔차'].abs().mean():>9.2f}{s['잔차'].mean():>10.2f}")
    print("  " + "-" * 44)
    mind = v.groupby("지평")["대상일"].nunique().min() if not v.empty else 0
    print(f"  · 최소 관측일수 : {mind}일")
    print("  · 편의보정 적용 요건 : 지평별 7일 이상(최소), 30일 이상 권장")
    print(f"  · 현 상태 : {'요건 충족' if mind >= 7 else '이력 부족 — 축적 필요'}")


def main():
    ap = argparse.ArgumentParser(description="주간 예측 이력 누적")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--summary", action="store_true", help="지평별 현황 출력")
    args = ap.parse_args()

    print("=" * 70)
    print(" step6_weekly_hist — 주간 예측 이력 누적 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {args.target}")
    print("=" * 70)
    rebuild(args.target)
    if args.summary:
        summary(args.target)
    print("=" * 70)


if __name__ == "__main__":
    main()
