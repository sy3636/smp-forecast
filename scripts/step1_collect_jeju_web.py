# -*- coding: utf-8 -*-
"""
step1_collect_jeju_web.py — 제주 SMP 웹 보완 수집기 (v1.0)
==============================================================================
배경
  · 공공데이터포털 「계통한계가격 및 수요예측」 API에서 **제주 행이 누락**되는 일자 발생
    - 2026. 8. 18. 이후 8일(8.18·8.22~24·8.30·8.31·9.6·9.7) — 일·월요일 반복
    - 원문 재수집 불가(응답 자체에 제주 행 부재), 공표 중단 공지 없음
  · 전력거래소 웹(제주 하루전시장 SMP)에는 **동일 일자 자료가 정상 게시**됨을 확인
    (2026. 9. 7. 거래분 1~24시간 확인, 2026. 9. 7. 조회)
  · 따라서 API 결손을 웹 자료로 **보완 수집**하여 데이터 공백을 해소

설계 원칙
  · **보완 전용** — API 수집분이 있으면 그대로 두고, 결측일만 채움
  · 원문 보존 : 수집 응답을 data/raw/smp_jeju_web/YYYYMMDD.html 로 저장
  · 산출 : data/interim/jeju_smp_web.csv (date, hour_n, smp_jeju, 출처, 수집시각)
  · 검증 : 24시간 완비·수치 범위(0 초과 1,000 미만) 확인 후에만 적재
  · 예의 : 요청 간 간격(기본 1.5초) 준수, User-Agent 명시

엔드포인트 탐색
  · 날짜 파라미터가 문서화되어 있지 않아 **--probe 로 후보 조합을 시험**함
  · 성공 조합은 config/jeju_web_endpoint.json 에 저장되어 이후 실행에 재사용

실행
  python scripts\\step1_collect_jeju_web.py --probe --date 20260830
  python scripts\\step1_collect_jeju_web.py --start 20260818 --end 20260907
  python scripts\\step1_collect_jeju_web.py --missing        # 데이터셋 결측일 자동 산정

유의
  · 웹 자료 이용은 기관 이용약관 범위 내에서 수행 [확인 필요]
  · 자동 크론 등록은 약관 확인 후 판단 — 본 도구는 **수동 실행 전제**

버전 : v1.0 (2026-09-07)
"""
import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timedelta

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("[중단] requests 미설치")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "smp_jeju_web")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CONFIG = os.path.join(BASE_DIR, "config")
LOG_DIR = os.path.join(BASE_DIR, "logs")
EP_FILE = os.path.join(CONFIG, "jeju_web_endpoint.json")
OUT_CSV = os.path.join(INTERIM, "jeju_smp_web.csv")

#  HTTP 헤더는 latin-1 로 인코딩되므로 **ASCII 만 사용**(한글 금지)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36 "
      "KOMIPO-SMP-Research/1.0")
TIMEOUT = 20

#  후보 조합 — (URL, 날짜 파라미터명, 날짜 서식)
CANDIDATES = [
    ("https://www.kpx.or.kr/smpJeju.es", "mid=a10404080200", "searchDate", "%Y-%m-%d"),
    ("https://www.kpx.or.kr/smpJeju.es", "mid=a10404080200", "searchDate", "%Y%m%d"),
    ("https://www.kpx.or.kr/smpJeju.es", "mid=a10404080200", "sdate", "%Y-%m-%d"),
    ("https://www.kpx.or.kr/smpJeju.es", "mid=a10404080200", "sdate", "%Y%m%d"),
    ("https://www.kpx.or.kr/smpJeju.es", "mid=a10404080200", "searchDay", "%Y%m%d"),
    ("https://www.kpx.or.kr/smpJeju.es", "mid=a10404080200", "baseDate", "%Y%m%d"),
    ("https://new.kpx.or.kr/smpJeju.es", "mid=a10606080200", "searchDate", "%Y-%m-%d"),
]

_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<24} {str(값):<26} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def fetch(url, fixed, key, day, fmt):
    q = f"{fixed}&{key}={day.strftime(fmt)}"
    full = f"{url}?{q}"
    r = requests.get(full, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return full, r.text


def parse_page(html):
    """페이지에서 (표시일자, {시간: 값}) 추출

    · 표시일자 : 'YYYY년 MM월 DD일' 패턴
    · 시간대 값 : 표 내 24개 수치 — '1h'~'24h' 라벨 또는 순차 셀
    """
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", html)
    shown = (f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
             if m else None)

    vals = {}
    #  ① '1h' 형태 라벨과 인접 수치
    for h in range(1, 25):
        pat = rf"{h}\s*h[^0-9\-]{{0,80}}?(-?\d{{1,4}}\.\d{{1,2}})"
        mm = re.search(pat, html)
        if mm:
            vals[h] = float(mm.group(1))
    if len(vals) == 24:
        return shown, vals

    #  ② 표 셀 순차 추출 — 소수 2자리 수치 24개 연속
    cells = re.findall(r">\s*(-?\d{1,4}\.\d{2})\s*<", html)
    if len(cells) >= 24:
        seq = [float(x) for x in cells[:24]]
        return shown, {i + 1: v for i, v in enumerate(seq)}
    return shown, vals


def validate(day_str, shown, vals):
    """수집 결과 검증 — (판정, 사유)"""
    if len(vals) != 24:
        return False, f"시간대 불완전({len(vals)}/24)"
    if shown and shown != day_str:
        return False, f"표시일자 불일치(요청 {day_str} / 표시 {shown})"
    bad = [h for h, v in vals.items() if not (0 < v < 1000)]
    if bad:
        return False, f"값 범위 이탈 {bad[:5]}"
    return True, "정상"


def load_endpoint():
    if os.path.exists(EP_FILE):
        with open(EP_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("url"), d.get("fixed"), d.get("key"), d.get("fmt")
    return None


def save_endpoint(url, fixed, key, fmt):
    os.makedirs(CONFIG, exist_ok=True)
    with open(EP_FILE, "w", encoding="utf-8") as f:
        json.dump({"url": url, "fixed": fixed, "key": key, "fmt": fmt,
                   "확정일": datetime.now().strftime("%Y-%m-%d")},
                  f, ensure_ascii=False, indent=2)


def probe(day, sleep):
    """후보 조합 시험 — 요청 일자가 실제로 반영되는 조합 탐색"""
    print("\n■ ① 엔드포인트 탐색")
    day_str = day.strftime("%Y-%m-%d")
    ok_list = []
    for url, fixed, key, fmt in CANDIDATES:
        try:
            full, html = fetch(url, fixed, key, day, fmt)
        except Exception as e:
            rep(f"{key}({fmt})", "요청 실패", str(e)[:40])
            time.sleep(sleep)
            continue
        shown, vals = parse_page(html)
        ok, why = validate(day_str, shown, vals)
        rep(f"{key}({fmt})", "적합" if ok else "부적합",
            f"표시 {shown} / 시간대 {len(vals)} — {why}")
        if ok:
            ok_list.append((url, fixed, key, fmt))
        time.sleep(sleep)

    if not ok_list:
        print("\n  [판정] 적합 조합 없음 — 날짜 파라미터가 JS 처리 방식일 가능성")
        print("         조치 : 브라우저 개발자도구로 실제 요청 확인 후 CANDIDATES 보강")
        return None
    save_endpoint(*ok_list[0])
    print(f"\n  [판정] 채택 : {ok_list[0][2]}({ok_list[0][3]}) → {EP_FILE}")
    return ok_list[0]


def missing_days():
    """데이터셋에서 육지 정상·제주 결측인 일자 산정"""
    p = os.path.join(PROCESSED, "smp_dataset_hourly.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 데이터셋 없음 → {p}")
    d = pd.read_csv(p, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c in ("date", "smp_land", "smp_jeju"))
    g = d.groupby("date").agg(land=("smp_land", "count"),
                              jeju=("smp_jeju", "count"))
    m = g[(g["land"] == 24) & (g["jeju"] < 24)]
    return [datetime.strptime(x, "%Y-%m-%d") for x in m.index]


def main():
    ap = argparse.ArgumentParser(description="제주 SMP 웹 보완 수집")
    ap.add_argument("--start", help="시작일 YYYYMMDD")
    ap.add_argument("--end", help="종료일 YYYYMMDD")
    ap.add_argument("--date", help="단일 일자 YYYYMMDD(탐색용)")
    ap.add_argument("--missing", action="store_true",
                    help="데이터셋 결측일 자동 산정하여 수집")
    ap.add_argument("--probe", action="store_true", help="엔드포인트 탐색만 수행")
    ap.add_argument("--sleep", type=float, default=1.5, help="요청 간격(초)")
    args = ap.parse_args()

    print("=" * 74)
    print(" step1_collect_jeju_web — 제주 SMP 웹 보완 수집 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    if args.probe:
        d = (datetime.strptime(args.date, "%Y%m%d") if args.date
             else datetime.now() - timedelta(days=1))
        probe(d, args.sleep)
        return

    ep = load_endpoint()
    if ep is None:
        d = (datetime.strptime(args.date, "%Y%m%d") if args.date
             else datetime.now() - timedelta(days=1))
        ep = probe(d, args.sleep)
        if ep is None:
            sys.exit("[중단] 엔드포인트 미확정 — --probe 결과 확인 후 재실행")
    url, fixed, key, fmt = ep

    #  대상 일자 산정
    if args.missing:
        days = missing_days()
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y%m%d")
        e = datetime.strptime(args.end, "%Y%m%d")
        days = [s + timedelta(days=i) for i in range((e - s).days + 1)]
    elif args.date:
        days = [datetime.strptime(args.date, "%Y%m%d")]
    else:
        sys.exit("[중단] --start/--end · --date · --missing 중 하나 지정")

    print("\n■ ① 대상")
    rep("엔드포인트", f"{key}({fmt})", os.path.basename(url))
    rep("대상 일자", f"{len(days)}일",
        f"{days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}" if days else "-")
    if not days:
        print("\n  [완료] 결측 일자 없음")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INTERIM, exist_ok=True)

    print("\n■ ② 수집")
    got, fail = [], []
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        try:
            full, html = fetch(url, fixed, key, d, fmt)
        except Exception as e:
            fail.append((ds, f"요청 실패 {str(e)[:30]}"))
            print(f"  [×] {ds} — 요청 실패")
            time.sleep(args.sleep)
            continue
        with open(os.path.join(RAW_DIR, f"{d:%Y%m%d}.html"), "w",
                  encoding="utf-8") as f:
            f.write(html)
        shown, vals = parse_page(html)
        ok, why = validate(ds, shown, vals)
        if ok:
            for h, v in sorted(vals.items()):
                got.append({"date": ds, "hour_n": h, "smp_jeju": v,
                            "출처": "KPX웹",
                            "수집시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            print(f"  [○] {ds} — 24시간 / 일평균 "
                  f"{sum(vals.values()) / 24:.2f}원")
        else:
            fail.append((ds, why))
            print(f"  [×] {ds} — {why}")
        time.sleep(args.sleep)

    print("\n■ ③ 산출")
    rep("수집 성공", f"{len(got) // 24}일", f"{len(got)}행")
    rep("수집 실패", f"{len(fail)}일",
        " / ".join(f"{d}:{w}" for d, w in fail[:3]) if fail else "-")

    if got:
        new = pd.DataFrame(got)
        if os.path.exists(OUT_CSV):
            old = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
            new = (pd.concat([old, new], ignore_index=True)
                   .sort_values(["date", "hour_n", "수집시각"])
                   .drop_duplicates(subset=["date", "hour_n"], keep="last"))
        new.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        rep("보완 파일", os.path.relpath(OUT_CSV, BASE_DIR), f"{len(new):,}행 누적")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"collect_jeju_web_{ts}.csv")
    pd.DataFrame(_rows).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("  ※ 데이터셋 반영은 별도 병합 절차 필요 — step2 보조원천 결합(후속 과제)")
    print("=" * 74)


if __name__ == "__main__":
    main()
