# -*- coding: utf-8 -*-
"""
check_na.py — 예측 대상일 입력 결측 변수 점검 (v1.0)
==============================================================================
목적
  · step6_predict 산출 로그의 「입력결측 N건」이 어느 변수에서 발생했는지 규명
  · 결측 원인(공표 지연·수집 누락·파생 실패)을 구분하여 조치 판단에 활용

실행
  python scripts\\check_na.py                          (익일 · 육지·제주)
  python scripts\\check_na.py --date 2026-09-05
  python scripts\\check_na.py --target smp_jeju --all  (부분 결측까지 표시)

버전 : v1.0 (2026-09-04)
"""
import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")


def check(t, day, show_all, variant=""):
    src = os.path.join(PROCESSED, f"smp_features_{t}{variant}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}{variant}.csv")
    if not os.path.exists(src):
        print(f"  [{t}] 피처셋 없음 → {src}")
        return
    d = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    seg = d[d["date"].astype(str) == day]
    print(f"\n■ {t}{variant} / 대상일 {day} — {len(seg)}행")
    if seg.empty:
        print("  [주의] 해당 일자 행 없음 — 피처셋 갱신 필요")
        return

    ok = None
    if os.path.exists(dic):
        dd = pd.read_csv(dic, encoding="utf-8-sig")
        ok = set(dd[dd["운영가용"] == "Y"]["변수명"])

    na_all = seg.isna().all()
    na_any = seg.isna().sum()
    full = [c for c in na_all.index if na_all[c]]
    part = [c for c in na_any.index if 0 < na_any[c] < len(seg)]

    def mark(c):
        return "" if ok is None else ("  [운영변수]" if c in ok else "")

    print(f"  · 전량 결측 : {len(full)}개")
    for c in full:
        print(f"      {c}{mark(c)}")
    if show_all and part:
        print(f"  · 부분 결측 : {len(part)}개")
        for c in part:
            print(f"      {c}  ({na_any[c]}/{len(seg)}행){mark(c)}")
    if ok is not None:
        blk = [c for c in full if c in ok]
        print(f"  · 운영가용 변수 중 전량 결측 : {len(blk)}개"
              + (f" — {', '.join(blk)}" if blk else " (없음)"))


def main():
    ap = argparse.ArgumentParser(description="예측 입력 결측 점검")
    ap.add_argument("--date", default=None, help="대상일 YYYY-MM-DD(기본 익일)")
    ap.add_argument("--target", default=None,
                    choices=["smp_land", "smp_jeju"], help="기본 양 계통")
    ap.add_argument("--all", action="store_true", help="부분 결측도 표시")
    ap.add_argument("--variant", default="",
                    help="변형 피처셋 접미사(예: _lagfill · _fuel) — A/B 실증 확인용")
    args = ap.parse_args()

    day = args.date or f"{datetime.now() + timedelta(days=1):%Y-%m-%d}"
    tg = [args.target] if args.target else ["smp_land", "smp_jeju"]

    print("=" * 70)
    print(" check_na — 예측 입력 결측 점검 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)
    for t in tg:
        check(t, day, args.all, args.variant)
    print("=" * 70)


if __name__ == "__main__":
    main()
