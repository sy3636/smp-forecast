# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 기상 데이터 수집(ASOS 종관기상관측)
================================================
목적 : 전력수요·SMP의 최대 설명변수인 기온을 확보하여 예측 정확도 개선
배경 : 현행 79변수 중 기상 변수 0건 — 냉난방 수요 변동을 달력·시차변수로만 간접 반영

설계
  · 조회 단위 : 지점 × 월 (1회 호출당 최대 744행) — 일별 조회 대비 호출량 1/30
  · 수집 지점 : 육지 7개 권역 대표지점 + 제주 1개 (기본값, --stations 로 변경 가능)
  · 재개 지원 : 기존 저장분 자동 건너뜀 — 중단 시 동일 명령 재실행
  · 트래픽    : 원장(logs/traffic_ledger.csv) 연동, 일 한도 초과 전 중단

입력 : config/endpoints.json (KMA_ASOS시간자료), config/.env
출력 : data/raw/asos/{지점번호}_{YYYYMM}.json
       logs/collect_asos_{일시}.csv (수집 이력)

버전 : v1.0 (2026-08-05)
실행 : python scripts\\step1_collect_weather.py --start 20210101 --end 20260731
       python scripts\\step1_collect_weather.py --start 20260701 --end 20260731 --stations 108,184
       python scripts\\step1_collect_weather.py --probe          (1건만 시험 호출)
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
from step1_collect import ledger_add, ledger_used, load_api, resolve_key  # noqa: E402

try:
    import requests
except ImportError:
    sys.exit("[중단] requests 미설치 → pip install -r requirements.txt")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "asos")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 관측지점 — 전력수요 권역 대표성 기준 선정
STATIONS = {
    "108": {"name": "서울", "region": "수도권", "weight": 0.30},
    "112": {"name": "인천", "region": "수도권", "weight": 0.10},
    "133": {"name": "대전", "region": "충청", "weight": 0.13},
    "143": {"name": "대구", "region": "경북", "weight": 0.13},
    "156": {"name": "광주", "region": "호남", "weight": 0.10},
    "159": {"name": "부산", "region": "경남", "weight": 0.17},
    "105": {"name": "강릉", "region": "강원", "weight": 0.07},
    "184": {"name": "제주", "region": "제주", "weight": 1.00},
}
LAND_IDS = [s for s in STATIONS if s != "184"]

DAILY_LIMIT = 10000     # ASOS 개발계정 일 한도(기상청 기준) [확인 필요]
SLEEP = 0.3             # 호출 간 대기(초)


# ------------------------------------------------------------------
def monthrange(start, end):
    """시작~종료월을 (년월 시작일, 종료일) 순으로 산출."""
    d = start.replace(day=1)
    while d <= end:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        s = max(d, start)
        e = min(nxt - timedelta(days=1), end)
        yield s, e
        d = nxt


def out_path(stn, ym):
    return os.path.join(RAW_DIR, f"{stn}_{ym}.json")


def fetch(url, params, key):
    """1회 호출. (성공여부, 건수, 메시지, 원문) 반환."""
    p = dict(params)
    p["serviceKey"] = key
    try:
        r = requests.get(url, params=p, timeout=TIMEOUT)
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {str(e)[:60]}", None
    if r.status_code == 429:
        return False, 0, "HTTP 429 트래픽 초과", None
    if r.status_code != 200:
        return False, 0, f"HTTP {r.status_code}", None
    try:
        j = r.json()
    except Exception:
        return False, 0, f"JSON 파싱 실패: {r.text[:60]}", None
    hdr = (j.get("response") or {}).get("header") or {}
    code = hdr.get("resultCode", "")
    if code not in ("00", "0"):
        return False, 0, f"결과코드 {code} {hdr.get('resultMsg', '')}", None
    body = (j.get("response") or {}).get("body") or {}
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    return True, len(items), f"{len(items)}건", j


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step1 ASOS 기상 데이터 수집")
    ap.add_argument("--start", default="20210101", help="시작일 YYYYMMDD")
    ap.add_argument("--end", default=None, help="종료일 YYYYMMDD(기본: 어제)")
    ap.add_argument("--stations", default=None,
                    help="지점번호 쉼표구분(기본: 정의된 8개 전체)")
    ap.add_argument("--probe", action="store_true", help="1건만 시험 호출 후 종료")
    ap.add_argument("--limit", type=int, default=DAILY_LIMIT, help="일 호출 한도")
    ap.add_argument("--force", action="store_true", help="기존 파일 있어도 재수집")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d")
    end = (datetime.strptime(args.end, "%Y%m%d") if args.end
           else datetime.now() - timedelta(days=1))
    stns = ([s.strip() for s in args.stations.split(",")] if args.stations
            else list(STATIONS))

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = load_env(ENV_PATH)
    api = load_api("ASOS")
    key = resolve_key(api, env)
    if not key:
        sys.exit("[중단] 인증키 없음 → config/.env 확인")

    print("=" * 78)
    print(" step1 ASOS 기상 데이터 수집 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 기간 : {start:%Y-%m-%d} ~ {end:%Y-%m-%d} / 지점 {len(stns)}개")
    print("=" * 78)

    months = list(monthrange(start, end))
    plan = [(s, m) for s in stns for m in months]
    used = ledger_used("asos")
    avail = max(0, args.limit - used)

    # 기존 저장분 확인
    todo = []
    for stn, (s, e) in plan:
        p = out_path(stn, f"{s:%Y%m}")
        if os.path.exists(p) and not args.force:
            continue
        todo.append((stn, s, e))

    print("\n■ 수집 계획")
    print(f"  · 대상 : {len(plan):,}건 (지점 {len(stns)} × {len(months)}개월)")
    print(f"  · 기수집 : {len(plan) - len(todo):,}건 (건너뜀)")
    print(f"  · 금회 수집 : {len(todo):,}건")
    print(f"  · 당일 사용 : {used:,}건 / 가용 {avail:,}건")

    if avail <= 0:
        sys.exit("\n[중단] 당일 트래픽 소진 → 익일 00시 이후 재실행")
    if len(todo) > avail:
        print(f"  · [주의] 가용량 초과 — {avail:,}건까지만 수집 후 중단")
        todo = todo[:avail]
    if not todo:
        print("\n[완료] 신규 수집 대상 없음 — 전량 확보 상태")
        return

    if args.probe:
        todo = todo[:1]
        print("\n  · --probe 지정 — 1건만 시험 호출")

    # ── 수집 ───────────────────────────────────
    print("\n■ 수집 진행")
    rows, ok_n, ng_n, calls = [], 0, 0, 0
    for i, (stn, s, e) in enumerate(todo, 1):
        p = dict(api["params"])
        p.update({"startDt": f"{s:%Y%m%d}", "endDt": f"{e:%Y%m%d}",
                  "startHh": "00", "endHh": "23",
                  "stnIds": stn, "numOfRows": "999", "pageNo": "1"})
        good, n, msg, raw = fetch(api["url"], p, key)
        calls += 1
        if good:
            with open(out_path(stn, f"{s:%Y%m}"), "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            ok_n += 1
        else:
            ng_n += 1
        rows.append({"지점": stn, "지점명": STATIONS.get(stn, {}).get("name", ""),
                     "년월": f"{s:%Y%m}", "판정": "정상" if good else "실패",
                     "건수": n, "비고": msg})

        if i % 10 == 0 or not good or i == len(todo):
            nm = STATIONS.get(stn, {}).get("name", stn)
            print(f"  [{i:>4}/{len(todo)}] {nm}({stn}) {s:%Y-%m} "
                  f"{'정상' if good else '실패'} — {msg}")
        if not good and "429" in msg:
            print("\n  [중단] 트래픽 초과 — 익일 재실행 시 미수집분부터 자동 재개")
            break
        time.sleep(SLEEP)

    ledger_add("asos", calls)

    # ── 결과 ───────────────────────────────────
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    lp = os.path.join(LOG_DIR, f"collect_asos_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["지점", "지점명", "년월", "판정", "건수", "비고"])
        w.writeheader()
        w.writerows(rows)

    total = len([1 for stn in stns for s, _ in months
                 if os.path.exists(out_path(stn, f"{s:%Y%m}"))])
    print("\n" + "=" * 78)
    print(f" 수집 결과 : 정상 {ok_n:,}건 / 실패 {ng_n:,}건 / 호출 {calls:,}회")
    print(f" 누적 확보 : {total:,}건 / 전체 {len(plan):,}건 "
          f"({total / len(plan) * 100:.1f}%)")
    print(f" 수집 이력 : {lp}")
    if total < len(plan):
        print(" ※ 미완료분은 동일 명령 재실행 시 자동 재개")
    else:
        print(" ※ 다음 단계 : python scripts\\step2_parse_weather.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
