# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 보조진단(원문 응답 확인)
================================================
목적 : 특정 API·일자의 원문 응답을 그대로 출력하여 0건 사유 규명
       (파라미터 오류 / 데이터 미보유 / 파싱 오류 구분)
버전 : v1.0 (2026-07-29)
실행 예:
  python scripts\\step1_raw_dump.py --api 태양광 --date 20250615
  python scripts\\step1_raw_dump.py --api 태양광 --date 2025-06-15
  python scripts\\step1_raw_dump.py --api 계통한계가격 --date 20260728
  python scripts\\step1_raw_dump.py --api 태양광 --date 20250615 --param tradeYmd
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import EP_PATH, ENV_PATH, load_env, TIMEOUT  # noqa: E402

import requests  # noqa: E402

DATE_PARAM_GUESS = {"태양광": "tradeYmd", "계통한계가격": "date"}


def main():
    ap = argparse.ArgumentParser(description="API 원문 응답 확인")
    ap.add_argument("--api", required=True, help="endpoints.json 의 API명 일부(예: 태양광)")
    ap.add_argument("--date", default=None, help="조회일자(YYYYMMDD 또는 YYYY-MM-DD)")
    ap.add_argument("--param", default=None, help="일자 파라미터명 강제 지정")
    ap.add_argument("--rows", default="10", help="numOfRows(기본 10)")
    ap.add_argument("--chars", type=int, default=1500, help="출력 문자수(기본 1500)")
    args = ap.parse_args()

    env = load_env(ENV_PATH)
    with open(EP_PATH, "r", encoding="utf-8") as f:
        apis = json.load(f).get("apis", [])

    api = next((a for a in apis if args.api in a.get("name", "")), None)
    if not api:
        sys.exit(f"[중단] '{args.api}' 포함 API 미정의")

    key = env.get(api.get("key_env", ""), "") or next(
        (v for k, v in env.items() if k.endswith("_API_KEY") and v), "")

    # 파라미터 구성(토큰값 제거 후 재지정)
    params = {k: v for k, v in api.get("params", {}).items()
              if not str(v).startswith("{")}
    params["numOfRows"] = args.rows
    params["pageNo"] = "1"

    if args.date:
        pname = args.param or next(
            (v for k, v in DATE_PARAM_GUESS.items() if k in api["name"]), "date")
        params[pname] = args.date

    params[api.get("key_param", "serviceKey")] = key

    masked = {k: ("****" if k.lower() == "servicekey" else v) for k, v in params.items()}
    print("=" * 70)
    print(f" API   : {api['name']}")
    print(f" URL   : {api['url']}")
    print(f" PARAM : {masked}")
    print("=" * 70)

    try:
        r = requests.get(api["url"], params=params, timeout=TIMEOUT)
    except Exception as e:
        # 예외 메시지에 인증키가 포함될 수 있으므로 마스킹 후 출력
        import urllib.parse
        msg = str(e)
        for form in (key, urllib.parse.quote(key, safe="")):
            if form:
                msg = msg.replace(form, "****")
        sys.exit(f"[실패] {type(e).__name__}: {msg}")

    print(f"HTTP   : {r.status_code}")
    print(f"길이   : {len(r.text)}자")
    print(f"헤더   : Content-Type={r.headers.get('Content-Type')}")
    print("-" * 70)
    print(r.text[:args.chars])
    if len(r.text) > args.chars:
        print(f"\n... (이하 {len(r.text) - args.chars}자 생략, --chars 로 확대)")


if __name__ == "__main__":
    main()
