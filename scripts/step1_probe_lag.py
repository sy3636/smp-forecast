# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 보조점검(데이터 가용시점 탐색)
================================================
목적 : 일자 기반 KPX API의 '최근 데이터 제공 시점(공표 지연일)' 확인
       - 계통한계가격 및 수요예측 : date
       - 지역별 시간별 태양광 발전량 : tradeYmd
배경 : 사전점검에서 태양광 API 조회건수 0건 → 정산 지연에 따른 미제공 여부 확인 필요
버전 : v1.0 (2026-07-29)
실행 : python scripts\\step1_probe_lag.py            (기본 15일 역순 탐색)
       python scripts\\step1_probe_lag.py --days 30
주의 : 개발계정 트래픽 100건/일 → --days 값 과다 설정 금지
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import (  # noqa: E402
    BASE_DIR, EP_PATH, ENV_PATH, load_env, parse_result, count_items,
    print_table, TIMEOUT,
)

import json  # noqa: E402
import requests  # noqa: E402

# 탐색 대상: (API명 키워드, 일자 파라미터명)
TARGETS = [
    ("계통한계가격", "date"),
    ("태양광", "tradeYmd"),
]


def probe(api, date_param, key, days):
    """일자를 하루씩 소급하며 데이터 보유 여부 확인."""
    rows = []
    first_hit = None
    for i in range(1, days + 1):
        d = f"{datetime.now() - timedelta(days=i):%Y%m%d}"
        params = {k: v for k, v in api.get("params", {}).items()
                  if not str(v).startswith("{")}
        params[date_param] = d
        params[api.get("key_param", "serviceKey")] = key
        try:
            r = requests.get(api["url"], params=params, timeout=TIMEOUT)
            code, msg = parse_result(r.text)
            cnt = count_items(r.text)
        except Exception as e:
            code, msg, cnt = "-", f"{type(e).__name__}", "-"
        rows.append({"조회일자": d, "결과코드": code or "-", "건수": cnt,
                     "비고": (msg or "")[:30]})
        if cnt not in ("-", "0", "") and code in ("00", "0", ""):
            first_hit = d
            break
    return rows, first_hit


def main():
    ap = argparse.ArgumentParser(description="KPX API 데이터 가용시점 탐색")
    ap.add_argument("--days", type=int, default=15, help="역순 탐색 일수(기본 15)")
    args = ap.parse_args()

    env = load_env(ENV_PATH)
    with open(EP_PATH, "r", encoding="utf-8") as f:
        apis = json.load(f).get("apis", [])

    print("=" * 70)
    print(" step1 보조점검 - 데이터 가용시점 탐색 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 최대 소급 {args.days}일")
    print("=" * 70)

    summary = []
    for kw, date_param in TARGETS:
        api = next((a for a in apis if kw in a.get("name", "")), None)
        if not api:
            print(f"\n[건너뜀] '{kw}' 대상 API 미정의")
            continue
        key = env.get(api.get("key_env", ""), "")
        if not key:
            key = next((v for k, v in env.items() if k.endswith("_API_KEY") and v), "")

        print(f"\n■ {api['name']}  (일자 파라미터: {date_param})")
        rows, hit = probe(api, date_param, key, args.days)
        print_table(rows, ["조회일자", "결과코드", "건수", "비고"])
        if hit:
            lag = (datetime.now() - datetime.strptime(hit, "%Y%m%d")).days
            print(f"  → 최근 보유일자 {hit} / 공표 지연 약 {lag}일")
            summary.append({"API명": api["name"], "최근보유일자": hit, "지연일": f"{lag}일"})
        else:
            print(f"  → 최근 {args.days}일 내 데이터 없음. 소급일수 확대 또는 활용신청 상태 확인 필요")
            summary.append({"API명": api["name"], "최근보유일자": "미확인",
                            "지연일": f">{args.days}일"})

    print("\nⅢ. 종합")
    print_table(summary, ["API명", "최근보유일자", "지연일"])
    print("\n※ 확인된 지연일을 step1 수집 스크립트의 기준일자 산정에 반영할 것")


if __name__ == "__main__":
    main()
