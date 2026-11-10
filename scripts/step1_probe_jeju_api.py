# -*- coding: utf-8 -*-
"""
step1_probe_jeju_api.py — 제주 행 누락 원인 진단 (v1.0)
==============================================================================
가설
  · 현행 수집은 `numOfRows=48` 고정 — 일자당 육지 24 + 제주 24 = 48행 전제
  · 응답 행 수가 48을 초과하는 일자에는 **제주 행이 페이지 경계에서 잘릴** 수 있음
  · 결측일에 육지 24행만 남은 현상과 정합

방법
  · 동일 일자를 numOfRows 48 · 200 두 조건으로 조회하여 areaName 분포 비교
  · 응답 totalCount 를 함께 출력하여 페이지 경계 여부 판정

실행
  python scripts\\step1_probe_jeju_api.py --date 20260830
  python scripts\\step1_probe_jeju_api.py --date 20260830 --date2 20260907

버전 : v1.0 (2026-09-07)
"""
import os
import sys
import json
import argparse
from datetime import datetime

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://apis.data.go.kr/B552115/SmpWithForecastDemand/"
       "getSmpWithForecastDemand")


def load_key():
    """인증키 확보 — 운영 수집기와 동일 경로(config/.env) 사용"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from step1_api_check import ENV_PATH, load_env
        env = load_env(ENV_PATH)
        k = env.get("KPX_API_KEY", "") or next(
            (v for kk, v in env.items() if kk.endswith("_API_KEY") and v), "")
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("KPX_API_KEY", "")


def query(key, day, rows):
    r = requests.get(URL, params={
        "serviceKey": key, "pageNo": "1", "numOfRows": str(rows),
        "dataType": "JSON", "date": day}, timeout=20)
    r.raise_for_status()
    try:
        d = r.json()
    except Exception:
        return None, r.text[:200]
    return d, None


def summarize(d):
    body = (d or {}).get("response", {}).get("body", {})
    total = body.get("totalCount")
    items = body.get("items", [])
    if isinstance(items, dict):
        items = items.get("item", [])
    areas = {}
    for it in items:
        a = str(it.get("areaName", "-"))
        areas[a] = areas.get(a, 0) + 1
    return total, len(items), areas


def main():
    ap = argparse.ArgumentParser(description="제주 행 누락 진단")
    ap.add_argument("--date", required=True, help="대상일 YYYYMMDD")
    ap.add_argument("--date2", help="대조일 YYYYMMDD(정상일)")
    args = ap.parse_args()

    key = load_key()
    if not key:
        sys.exit("[중단] KPX_API_KEY 미확보")

    print("=" * 70)
    print(" step1_probe_jeju_api — 제주 행 누락 진단 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    days = [args.date] + ([args.date2] if args.date2 else [])
    for day in days:
        print(f"\n■ 대상일 {day}")
        for rows in (48, 200):
            d, err = query(key, day, rows)
            if err:
                print(f"  numOfRows={rows:<4} 응답 오류 — {err}")
                continue
            total, n, areas = summarize(d)
            dist = " / ".join(f"{k} {v}행" for k, v in sorted(areas.items()))
            print(f"  numOfRows={rows:<4} totalCount={total} 수신={n}행  {dist or '없음'}")
    print("\n  ※ 48 조건에서 제주 부재·200 조건에서 제주 존재 → **페이지 경계 잘림 확정**")
    print("=" * 70)


if __name__ == "__main__":
    main()
