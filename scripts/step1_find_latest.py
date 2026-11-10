# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 보조진단(최신 보유일자 이분탐색)
================================================
목적 : 일자 기반 API의 '데이터 보유 최종일자'를 최소 호출로 확정
       (일별 순차조회 대비 호출수 1/20 수준 → 개발계정 트래픽 100건/일 보호)
방식 : 데이터 보유 구간이 연속이라는 전제하에 이분탐색(약 9~11회 호출)
버전 : v1.0 (2026-07-29)
실행 예:
  python scripts\\step1_find_latest.py --api 태양광
  python scripts\\step1_find_latest.py --api 태양광 --anchor 20250615
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import EP_PATH, ENV_PATH, load_env, TIMEOUT  # noqa: E402

import requests  # noqa: E402

DATE_PARAM_GUESS = {"태양광": "tradeYmd", "계통한계가격": "date"}


def fetch_count(api, key, date_param, d, calls):
    """지정 일자의 totalCount 반환(호출 카운터 증가)."""
    params = {k: v for k, v in api.get("params", {}).items()
              if not str(v).startswith("{")}
    params.update({"pageNo": "1", "numOfRows": "1", date_param: d})
    params[api.get("key_param", "serviceKey")] = key
    calls[0] += 1
    try:
        r = requests.get(api["url"], params=params, timeout=TIMEOUT)
        js = json.loads(r.text)
        body = js.get("response", {}).get("body", {})
        return int(str(body.get("totalCount", "0")).strip() or 0)
    except Exception:
        return -1  # 오류


def main():
    ap = argparse.ArgumentParser(description="데이터 보유 최종일자 이분탐색")
    ap.add_argument("--api", required=True, help="API명 일부(예: 태양광)")
    ap.add_argument("--anchor", default=None,
                    help="데이터 보유가 확인된 기준일자(YYYYMMDD). 미지정 시 자동 소급 탐색")
    ap.add_argument("--param", default=None, help="일자 파라미터명 강제 지정")
    args = ap.parse_args()

    env = load_env(ENV_PATH)
    with open(EP_PATH, "r", encoding="utf-8") as f:
        apis = json.load(f).get("apis", [])
    api = next((a for a in apis if args.api in a.get("name", "")), None)
    if not api:
        sys.exit(f"[중단] '{args.api}' 포함 API 미정의")

    key = env.get(api.get("key_env", ""), "") or next(
        (v for k, v in env.items() if k.endswith("_API_KEY") and v), "")
    date_param = args.param or next(
        (v for k, v in DATE_PARAM_GUESS.items() if k in api["name"]), "date")

    calls = [0]
    today = datetime.now().date()

    print("=" * 70)
    print(f" 최신 보유일자 탐색 : {api['name']}")
    print(f" 일자 파라미터      : {date_param}")
    print("=" * 70)

    # (1) 하한(데이터 보유 확인일) 설정
    if args.anchor:
        lo = datetime.strptime(args.anchor, "%Y%m%d").date()
        if fetch_count(api, key, date_param, f"{lo:%Y%m%d}", calls) <= 0:
            sys.exit(f"[중단] 기준일자 {lo:%Y%m%d} 에 데이터 없음 → --anchor 소급 재지정 필요")
    else:
        lo, step = None, 30
        probe = today - timedelta(days=step)
        while step <= 480:
            if fetch_count(api, key, date_param, f"{probe:%Y%m%d}", calls) > 0:
                lo = probe
                break
            step *= 2
            probe = today - timedelta(days=step)
        if lo is None:
            sys.exit(f"[중단] 최근 480일 내 보유 데이터 미확인(호출 {calls[0]}회)")
    print(f"  하한 확정 : {lo:%Y%m%d} (데이터 보유)")

    # (2) 이분탐색 : 보유 최종일자
    hi = today
    while (hi - lo).days > 1:
        mid = lo + timedelta(days=(hi - lo).days // 2)
        cnt = fetch_count(api, key, date_param, f"{mid:%Y%m%d}", calls)
        mark = "보유" if cnt > 0 else ("오류" if cnt < 0 else "없음")
        print(f"  {mid:%Y%m%d} → {mark}(totalCount={cnt})")
        if cnt > 0:
            lo = mid
        else:
            hi = mid

    lag = (today - lo).days
    print("-" * 70)
    print(f" 결과 : 보유 최종일자 {lo:%Y%m%d} / 공표 지연 약 {lag}일 / 총 호출 {calls[0]}회")
    print(f" 반영 : step1 수집 기준일 = 실행일 - {lag}일")


if __name__ == "__main__":
    main()
