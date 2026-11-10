# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 데이터 수집
================================================
목적 : 학습·백테스트용 원천데이터를 일자 단위로 수집하여 data/raw 에 원문 보존
대상 : ① SMP·수요예측(일별) ② 지역별 시간별 태양광 발전량(일별) ③ 공휴일(월별)
원칙 : 원문 무가공 저장(재현성 확보), 중복 수집 방지, 트래픽 한도 준수
버전 : v1.0 (2026-07-29)

실행 예:
  # 잔여 수집대상 확인(호출 없음)
  python scripts\\step1_collect.py --source smp --start 20230101 --end 20260728 --plan
  # 실제 수집(1회 최대 90건 → 트래픽 100건/일 한도 내)
  python scripts\\step1_collect.py --source smp --start 20230101 --end 20260728
  # 재실행 시 미수집분부터 자동 재개
  python scripts\\step1_collect.py --source pv --start 20230101 --end 20260609
  python scripts\\step1_collect.py --source holiday --start 20230101 --end 20261231

주의 : 개발계정 트래픽 SMP·태양광 각 100건/일 → 전 기간 수집에 다일 소요
       (2023.1.1~2026.7.28 = 1,306일 → 일 90건 기준 약 15일)
       운영계정 승인 시 --max-calls 상향 조정
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR, EP_PATH, ENV_PATH, load_env, TIMEOUT  # noqa: E402

import requests  # noqa: E402

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LEDGER = os.path.join(LOG_DIR, "traffic_ledger.csv")  # 일자·수집원별 호출 누계

# 수집원 정의
SOURCES = {
    "smp": {
        "api_kw": "계통한계가격",
        "mode": "daily",
        "date_param": "date",
        "rows": "100",          # 1일 48건
        "expect": 48,           # 완전 수집 기준 — 미만이면 재수집(육지 24 + 제주 24)
        "ext": "json",
        "desc": "SMP·수요예측(육지·제주 24시간)",
    },
    "pv": {
        "api_kw": "태양광",
        "mode": "daily",
        "date_param": "tradeYmd",
        "rows": "500",          # 1일 408건(17개 시·도 × 24시간)
        "ext": "json",
        "desc": "지역별 시간별 태양광 발전량",
    },
    "holiday": {
        "api_kw": "특일정보",
        "mode": "monthly",
        "rows": "50",
        "ext": "xml",
        "desc": "공휴일(월별 조회)",
    },
}


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def monthrange(start, end):
    d = start.replace(day=1)
    while d <= end:
        yield d
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def load_api(kw):
    with open(EP_PATH, "r", encoding="utf-8") as f:
        apis = json.load(f).get("apis", [])
    api = next((a for a in apis if kw in a.get("name", "")), None)
    if not api:
        sys.exit(f"[중단] '{kw}' 포함 API 미정의 → config/endpoints.json 확인")
    return api


def resolve_key(api, env):
    return env.get(api.get("key_env", ""), "") or next(
        (v for k, v in env.items() if k.endswith("_API_KEY") and v), "")


def ledger_used(src):
    """당일 해당 수집원의 누적 호출수 조회."""
    if not os.path.exists(LEDGER):
        return 0
    today = f"{datetime.now():%Y%m%d}"
    total = 0
    with open(LEDGER, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("일자") == today and row.get("수집원") == src:
                total += int(row.get("호출수", 0) or 0)
    return total


def ledger_add(src, n):
    """당일 호출수 기록."""
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["일자", "시각", "수집원", "호출수"])
        if new:
            w.writeheader()
        w.writerow({"일자": f"{datetime.now():%Y%m%d}",
                    "시각": f"{datetime.now():%H:%M:%S}",
                    "수집원": src, "호출수": n})


def is_valid(text, ext):
    """수신 응답의 유효성 판정. (정상여부, 건수, 메시지)"""
    if ext == "json":
        try:
            js = json.loads(text)
            hdr = js.get("response", {}).get("header", {})
            body = js.get("response", {}).get("body", {})
            code = str(hdr.get("resultCode", "")).zfill(2)
            cnt = int(str(body.get("totalCount", "0")).strip() or 0)
            return code == "00", cnt, hdr.get("resultMsg", "")
        except Exception as e:
            return False, 0, f"파싱오류:{type(e).__name__}"
    # XML
    ok = "<resultCode>00</resultCode>" in text.replace(" ", "")
    cnt = text.count("<item>")
    return ok, cnt, "OK" if ok else "결과코드 비정상"


# ------------------------------------------------------------------
# 수집 본체
# ------------------------------------------------------------------
def collect(src, start, end, max_calls, plan_only, sleep_sec, quota=100, reserve=5, force=False):
    cfg = SOURCES[src]
    env = load_env(ENV_PATH)
    api = load_api(cfg["api_kw"])
    key = resolve_key(api, env)
    if not key:
        sys.exit("[중단] 인증키 없음 → config/.env 확인")

    out_dir = os.path.join(RAW_DIR, src)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 수집 단위 목록 생성
    if cfg["mode"] == "daily":
        units = [f"{d:%Y%m%d}" for d in daterange(start, end)]
    else:
        units = [f"{d:%Y%m}" for d in monthrange(start, end)]

    #  재수집 판정 (2026.9.7 개정)
    #   · 종전 : 파일 존재 시 무조건 건너뜀
    #   · 결함 : 제주 공표가 육지보다 늦은 일자에 **육지 24행만 저장된 원문이 캐시**되어
    #            이후 API가 48행으로 갱신되어도 재수집되지 않음
    #            (2026.8.18 이후 8일 — 제주 결측 288건의 실제 원인)
    #   · 개정 : 저장된 원문이 **불완전(기대 행수 미만)** 하면 재수집 대상에 포함
    #   · --force 지정 시 기간 내 전량 재수집
    def _incomplete(path):
        if cfg.get("expect") is None or not path.endswith(".json"):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                b = json.load(f).get("response", {}).get("body", {})
            return int(b.get("totalCount", 0)) < int(cfg["expect"])
        except Exception:
            return True

    todo = []
    for u in units:
        fp = os.path.join(out_dir, f"{u}.{cfg['ext']}")
        if force or not os.path.exists(fp) or _incomplete(fp):
            todo.append(u)
    done_cnt = len(units) - len(todo)

    used = ledger_used(src)
    budget = max(0, quota - used - reserve)   # 진단용 여유분 확보
    limit = min(max_calls, budget)

    print("=" * 70)
    print(f" step1 데이터 수집 : {src} ({cfg['desc']})")
    print(f" 기간   : {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")
    print(f" 대상   : 전체 {len(units)}건 / 기수집 {done_cnt}건 / 잔여 {len(todo)}건")
    print(f" 트래픽 : 한도 {quota} / 당일 사용 {used} / 예비 {reserve} → 가용 {budget}건")
    print(f" 이번회 : 최대 {limit}건 호출")
    print("=" * 70)

    if not plan_only and limit <= 0:
        print(" [중단] 당일 트래픽 소진 → 익일 재실행 필요")
        return

    if plan_only:
        if todo:
            print(f" 잔여 시작점 : {todo[0]} / 종료점 : {todo[-1]}")
            days = (len(todo) + max_calls - 1) // max_calls
            print(f" 예상 소요   : 약 {days}일(1일 {max_calls}건 기준)")
        else:
            print(" 전량 수집 완료 상태")
        return

    if not todo:
        print(" [완료] 잔여 없음")
        return

    log_rows = []
    ok_cnt = ng_cnt = empty_cnt = 0
    for i, unit in enumerate(todo[:limit], 1):
        params = {k: v for k, v in api.get("params", {}).items()
                  if not str(v).startswith("{")}
        params["pageNo"] = "1"
        params["numOfRows"] = cfg["rows"]
        if cfg["mode"] == "daily":
            params[cfg["date_param"]] = unit
        else:
            params["solYear"] = unit[:4]
            params["solMonth"] = unit[4:]
        params[api.get("key_param", "serviceKey")] = key

        status, cnt, msg = "실패", 0, ""
        try:
            r = requests.get(api["url"], params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                ok, cnt, msg = is_valid(r.text, cfg["ext"])
                if ok and cnt > 0:
                    path = os.path.join(out_dir, f"{unit}.{cfg['ext']}")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(r.text)
                    status, ok_cnt = "정상", ok_cnt + 1
                elif ok:
                    status, empty_cnt = "무데이터", empty_cnt + 1
                else:
                    ng_cnt += 1
            elif r.status_code == 429:
                ng_cnt += 1
                log_rows.append({"수집단위": unit, "상태": "실패", "건수": 0,
                                 "비고": "HTTP 429 일일 트래픽 초과"})
                print(f"  [즉시중단] {unit} → 일일 트래픽 한도 초과(HTTP 429)")
                break
            else:
                msg, ng_cnt = f"HTTP {r.status_code}", ng_cnt + 1
        except Exception as e:
            msg, ng_cnt = f"{type(e).__name__}", ng_cnt + 1

        log_rows.append({"수집단위": unit, "상태": status, "건수": cnt, "비고": msg[:40]})
        if i % 10 == 0 or i == len(todo[:limit]):
            print(f"  진행 {i}/{min(limit, len(todo))}  최근 {unit} → {status}({cnt}건)")

        # 연속 실패 시 조기 중단(트래픽 소모 방지)
        if ng_cnt >= 5 and ok_cnt == 0:
            print("  [중단] 연속 실패 5회 → 인증키·승인상태 확인 필요")
            break
        time.sleep(sleep_sec)

    # 트래픽 원장 기록
    ledger_add(src, len(log_rows))

    # 결과 기록
    log_path = os.path.join(LOG_DIR, f"collect_{src}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["수집단위", "상태", "건수", "비고"])
        w.writeheader()
        w.writerows(log_rows)

    remain = len(todo) - len(log_rows)
    print("-" * 70)
    print(f" 결과 : 정상 {ok_cnt} / 무데이터 {empty_cnt} / 실패 {ng_cnt}")
    print(f" 저장 : {out_dir}")
    print(f" 로그 : {log_path}")
    if remain > 0:
        print(f" 잔여 {remain}건 → 익일 동일 명령 재실행 시 자동 재개")
    else:
        print(" 전 기간 수집 완료")


def main():
    ap = argparse.ArgumentParser(description="step1 원천데이터 수집")
    ap.add_argument("--source", required=True, choices=list(SOURCES) + ["all"])
    ap.add_argument("--start", required=True, help="시작일 YYYYMMDD")
    ap.add_argument("--end", required=True, help="종료일 YYYYMMDD")
    ap.add_argument("--max-calls", type=int, default=100, help="1회 최대 호출수(기본 100)")
    ap.add_argument("--quota", type=int, default=100, help="일일 트래픽 한도(기본 100)")
    ap.add_argument("--reserve", type=int, default=5, help="진단용 예비 호출수(기본 5)")
    ap.add_argument("--sleep", type=float, default=0.3, help="호출 간 대기(초)")
    ap.add_argument("--force", action="store_true",
                    help="기수집분 포함 전량 재수집(불완전 원문 갱신용)")
    ap.add_argument("--plan", action="store_true", help="계획만 출력(호출 없음)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()
    if start > end:
        sys.exit("[중단] 시작일이 종료일보다 늦음")

    targets = list(SOURCES) if args.source == "all" else [args.source]
    for src in targets:
        collect(src, start, end, args.max_calls, args.plan, args.sleep,
                args.quota, args.reserve, args.force)
        print()


if __name__ == "__main__":
    main()
