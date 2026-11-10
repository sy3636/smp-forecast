# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 일괄수집(페이지네이션 방식)
================================================
배경 : 일자 파라미터 생략 시 전 기간 데이터가 단일 질의로 제공됨(totalCount 115,344 확인)
       → 일별 조회(1일 1호출) 대비 호출수를 수십 분의 1로 축소
목적 : numOfRows 단위 페이지 순회로 전 기간 원문을 일괄 확보
출력 : data/raw/{source}_bulk/p0001.json … (페이지 원문)
버전 : v1.0 (2026-07-30)

실행 :
  # 1) numOfRows 상한 시험(1호출)
  python scripts\\step1_collect_bulk.py --source smp --probe --rows 10000
  # 2) 일괄 수집
  python scripts\\step1_collect_bulk.py --source smp --rows 10000
  # 3) 중단 시 재실행(기수집 페이지 자동 건너뜀)
  python scripts\\step1_collect_bulk.py --source smp --rows 10000

주의 : 데이터는 최신일자 우선 정렬로 반환됨. 일 1회(23시경) 갱신되므로
       갱신 시각을 걸쳐 수집할 경우 페이지 경계 어긋남 발생 가능
       → 1개 세션 내 완료 권장, 완료 후 step2 정합성 검증 필수
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR, EP_PATH, ENV_PATH, load_env, TIMEOUT  # noqa: E402
from step1_collect import ledger_add, ledger_used  # noqa: E402

import requests  # noqa: E402

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

SOURCES = {
    "smp": {"api_kw": "계통한계가격", "desc": "SMP·수요예측"},
    "pv": {"api_kw": "태양광", "desc": "지역별 시간별 태양광 발전량"},
}


def call(api, key, page, rows):
    """단일 페이지 호출. (응답본문, totalCount, 건수) 반환."""
    params = {k: v for k, v in api.get("params", {}).items()
              if not str(v).startswith("{")}
    params.update({"pageNo": str(page), "numOfRows": str(rows), "dataType": "JSON"})
    # 일자 파라미터는 제외(전 기간 조회)
    for p in ("date", "tradeYmd"):
        params.pop(p, None)
    params[api.get("key_param", "serviceKey")] = key

    r = requests.get(api["url"], params=params, timeout=TIMEOUT * 4)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    js = json.loads(r.text)
    body = js.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    return r.text, int(str(body.get("totalCount", "0")).strip() or 0), len(items)


def main():
    ap = argparse.ArgumentParser(description="step1 일괄수집(페이지네이션)")
    ap.add_argument("--source", default="smp", choices=list(SOURCES))
    ap.add_argument("--rows", type=int, default=10000, help="페이지당 건수(기본 10000)")
    ap.add_argument("--probe", action="store_true", help="1페이지만 호출하여 상한 확인")
    ap.add_argument("--max-calls", type=int, default=100, help="1회 최대 호출수")
    ap.add_argument("--quota", type=int, default=100, help="일일 트래픽 한도")
    ap.add_argument("--reserve", type=int, default=2, help="예비 호출수")
    ap.add_argument("--sleep", type=float, default=1.0, help="호출 간 대기(초)")
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    env = load_env(ENV_PATH)
    with open(EP_PATH, "r", encoding="utf-8") as f:
        apis = json.load(f).get("apis", [])
    api = next((a for a in apis if cfg["api_kw"] in a.get("name", "")), None)
    if not api:
        sys.exit(f"[중단] '{cfg['api_kw']}' 포함 API 미정의")
    key = env.get(api.get("key_env", ""), "") or next(
        (v for k, v in env.items() if k.endswith("_API_KEY") and v), "")

    out_dir = os.path.join(RAW_DIR, f"{args.source}_bulk")
    os.makedirs(out_dir, exist_ok=True)

    used = ledger_used(args.source)
    budget = max(0, args.quota - used - args.reserve)

    print("=" * 70)
    print(f" step1 일괄수집 : {args.source} ({cfg['desc']})")
    print(f" 페이지당 건수 : {args.rows:,}")
    print(f" 트래픽 : 한도 {args.quota} / 당일 사용 {used} / 예비 {args.reserve}"
          f" → 가용 {budget}건")
    print("=" * 70)

    # ── 상한 시험 ────────────────────────────────
    if args.probe:
        if budget < 1:
            sys.exit(" [중단] 가용 호출 없음")
        text, total, n = call(api, key, 1, args.rows)
        ledger_add(args.source, 1)
        print(f"  요청 numOfRows : {args.rows:,}")
        print(f"  실제 수신 건수 : {n:,}")
        print(f"  전체 건수      : {total:,}")
        if n < args.rows and n < total:
            print(f"  → 서버 상한 {n:,}건으로 제한됨. --rows {n} 사용 권장")
            pages = math.ceil(total / n) if n else 0
        else:
            print("  → 요청값 그대로 수신. 상한 미도달")
            pages = math.ceil(total / args.rows)
        print(f"  예상 소요 페이지 : {pages:,}회 호출")
        return

    # ── 본 수집 ──────────────────────────────────
    if budget < 1:
        sys.exit(" [중단] 당일 트래픽 소진 → 익일 재실행")

    # 페이지 크기 일관성 보장(혼용 시 구간 누락 발생)
    meta_path = os.path.join(out_dir, "_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if int(meta.get("rows", 0)) != args.rows:
            sys.exit(f" [중단] 기존 수집분 페이지크기 {meta['rows']:,} ≠ 요청 {args.rows:,}\n"
                     f"        → --rows {meta['rows']} 로 실행하거나, "
                     f"{out_dir} 폴더 삭제 후 재수집")
    else:
        json.dump({"rows": args.rows, "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}"},
                  open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)

    limit = min(args.max_calls, budget)
    page, calls, got = 1, 0, 0
    total = None
    t0 = time.time()

    while calls < limit:
        path = os.path.join(out_dir, f"p{page:04d}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
            body = js.get("response", {}).get("body", {})
            total = total or int(str(body.get("totalCount", "0")).strip() or 0)
            it = body.get("items", {}).get("item", [])
            got += len(it) if isinstance(it, list) else 1
            page += 1
            continue

        try:
            text, total, n = call(api, key, page, args.rows)
        except Exception as e:
            print(f"  [중단] p{page:04d} 실패 : {type(e).__name__} {str(e)[:60]}")
            break
        calls += 1
        if n == 0:
            print(f"  p{page:04d} 수신 0건 → 종료")
            break
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        got += n
        print(f"  p{page:04d} 저장 {n:,}건 (누계 {got:,}/{total:,})")
        if n < args.rows and got < total:
            print(f"  [경고] 요청 {args.rows:,}건 대비 {n:,}건만 수신 → 서버 상한 추정.\n"
                  f"        {out_dir} 폴더 삭제 후 --rows {n} 로 재수집 필요")
            break
        page += 1
        if got >= total:
            break
        time.sleep(args.sleep)

    ledger_add(args.source, calls)
    print("-" * 70)
    print(f" 호출 {calls}회 / 확보 {got:,}건 / 전체 {total:,}건 "
          f"/ 소요 {time.time() - t0:.1f}초")
    if total and got >= total:
        print(" [완료] 전 기간 확보 → python scripts\\step2_preprocess.py 실행")
    else:
        print(" 잔여분 존재 → 동일 명령 재실행 시 자동 재개")


if __name__ == "__main__":
    main()
