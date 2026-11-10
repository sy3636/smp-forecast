# -*- coding: utf-8 -*-
"""
step1_collect_kpx_web.py — SMP·수요예측 전력거래소 웹 보완 수집기 (v1.0, 수동 실행)
==============================================================================
배경
  · 2026. 10. 1.~2. 공공데이터포털 「계통한계가격 및 수요예측」 API에 10. 2.분이 0건
    (10. 1. 18:00·21:00·23:30, 10. 2. 06:43 확인) — 전력거래소 홈페이지에는 정상 공표
  · 영향 : 실적 미적재(예측 프로그램·Grafana 공란), D+1 수요예측 부재로 폴백 산출 연속
  · 기존 제주 웹 수집기(step1_collect_jeju_web.py)는 날짜 조회 주소 미확정·파이프라인 미연결

동작
  ① 홈페이지 4쪽 조회 — 육지·제주 SMP, 육지·제주 하루전 발전계획용 수요예측
     (각 쪽은 기본으로 최근 7일 거래분을 표시 — 날짜 파라미터 불요)
  ② 대조 검증 — API 원문이 있는 날은 웹 값과 전 항목 대조, **1건이라도 불일치 시 저장 중단**
  ③ 보완 대상 — API 원문(data/raw/smp/YYYYMMDD.json)이 없거나 48행 미만인 날만
  ④ 저장 — API와 같은 구조로 data/raw/smp_web/YYYYMMDD.json (source=kpx_web)
           24시간 완비·값 범위(SMP 0~1,000원, 수요예측 >0) 확인 후에만 저장
  ⑤ 반영 — step2_preprocess 가 API 원문 우선, 빈 날만 웹 원문 사용(자동)

원칙
  · **수동 실행 전용**(2026. 10. 2. 부장 결정) — 웹 자료 자동 수집은 기관 이용약관 확인 후 판단
  · API 원문은 건드리지 않음 — API가 따라잡으면 API 값이 자동 우선
  · 원본 HTML 보존 : data/raw/smp_web/html/{쪽}_{일시}.html
  · 요청 간 1.5초 간격, User-Agent 명시

실행 (서버)
  docker exec -w /app smp-forecast python scripts/step1_collect_kpx_web.py --dry-run
  docker exec -w /app smp-forecast python scripts/step1_collect_kpx_web.py
  → 이후 반영 : 운영매뉴얼 Ⅵ-V 「API 공표 지연」 절차

버전 : v1.0 (2026-10-02)
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime

try:
    import requests
except ImportError:
    sys.exit("[중단] requests 미설치")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_API = os.path.join(BASE_DIR, "data", "raw", "smp")
RAW_WEB = os.path.join(BASE_DIR, "data", "raw", "smp_web")
HTML_DIR = os.path.join(RAW_WEB, "html")
LOG_DIR = os.path.join(BASE_DIR, "logs")

#  HTTP 헤더는 latin-1 인코딩 — ASCII만 사용
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 SMP-forecast-research")
PAGES = {   # 키 : (URL, 원문 항목)
    "smp_land": ("https://new.kpx.or.kr/smpInland.es?mid=a10606080100&device=pc", "smp/육지"),
    "smp_jeju": ("https://new.kpx.or.kr/smpJeju.es?mid=a10606080200&device=pc", "smp/제주"),
    "pss_land": ("https://new.kpx.or.kr/pssInland.es?mid=a10606090100&device=pc", "mlfd"),
    "pss_jeju": ("https://new.kpx.or.kr/pssJeju.es?mid=a10606090200&device=pc", "jlfd"),
}
SLEEP = 1.5
TIMEOUT = 20
EXPECT = 48          # 원문 1일 기대 행수 — 육지 24 + 제주 24

_rows = []


def rep(항목, 값, 비고=""):
    print("  · {:<22} {:<28} {}".format(항목, 값, 비고))
    _rows.append({"항목": 항목, "값": 값, "비고": 비고})


def _num(s):
    s = re.sub(r"<[^>]+>", "", s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_table(html):
    """첫 conTable → {YYYYMMDD: [24값]} — 연도는 「YYYY년 MM월 DD일 거래분」 기준 추정."""
    m = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*거래분", html)
    if not m:
        raise ValueError("거래분 표기 없음")
    end = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    t = re.search(r"<table[^>]*conTable[^>]*>(.*?)</table>", html, re.S)
    if not t:
        raise ValueError("표 없음")
    body = t.group(1)
    heads = re.findall(r"<th[^>]*>\s*(\d{2})\.(\d{2})\s*<br", body)
    days = []
    for mm, dd in heads:
        y = end.year - (1 if int(mm) > end.month else 0)        # 연말·연초 경계
        days.append(date(y, int(mm), int(dd)).strftime("%Y%m%d"))
    out = {d: [None] * 24 for d in days}
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if not tds:
            continue
        h = re.match(r"\s*(\d{1,2})h\s*$", re.sub(r"<[^>]+>", "", tds[0]))
        if not h:
            continue                                  # 최대·최소·가중평균 행 제외
        hi = int(h.group(1)) - 1
        for d, v in zip(days, tds[1:]):
            if 0 <= hi < 24:
                out[d][hi] = _num(v)
    return end.strftime("%Y%m%d"), out


def load_api(day):
    """API 원문 → {(areaName, hour): item} / 없으면 None."""
    p = os.path.join(RAW_API, "{}.json".format(day))
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        js = json.load(f)
    items = js.get("response", {}).get("body", {}).get("items", {}) or {}
    items = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(items, dict):
        items = [items]
    return {(i.get("areaName"), int(i.get("hour"))): i for i in items}


def build_items(day, web):
    items = []
    for area, key in (("육지", "smp_land"), ("제주", "smp_jeju")):
        for h in range(24):
            ml, jl = web["pss_land"][day][h], web["pss_jeju"][day][h]
            items.append({"date": day, "hour": "{:02d}".format(h + 1), "areaName": area,
                          "smp": web[key][day][h], "mlfd": ml, "jlfd": jl,
                          "slfd": round(ml + jl, 1), "source": "kpx_web"})
    return items


def valid(day, web):
    probs = []
    for k in PAGES:
        v = web[k].get(day)
        if v is None or any(x is None for x in v):
            probs.append("{} 24시간 미완비".format(k))
            continue
        if k.startswith("smp") and any(not (0 <= x < 1000) for x in v):
            probs.append("{} 범위 이탈".format(k))
        if k.startswith("pss") and any(x <= 0 for x in v):
            probs.append("{} 0 이하".format(k))
    return probs


def main():
    ap = argparse.ArgumentParser(description="SMP·수요예측 전력거래소 웹 보완 수집(수동)")
    ap.add_argument("--dry-run", action="store_true", help="조회·대조만 — 저장 안 함")
    ap.add_argument("--dates", default="", help="보완 대상 제한 YYYYMMDD,... (기본 : API 원문 공백일 전체)")
    a = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 78)
    print(" step1_collect_kpx_web — SMP·수요예측 웹 보완 수집 v1.0 (수동)")
    print(" 실행 : {}{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            " / DRY-RUN" if a.dry_run else ""))
    print("=" * 78)

    print("\n■ ① 홈페이지 조회")
    web, ends = {}, set()
    os.makedirs(HTML_DIR, exist_ok=True)
    for i, (k, (url, _)) in enumerate(PAGES.items()):
        if i:
            time.sleep(SLEEP)
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1"):
            r.encoding = "utf-8"
        if not a.dry_run:
            with open(os.path.join(HTML_DIR, "{}_{}.html".format(k, ts)), "w", encoding="utf-8") as f:
                f.write(r.text)
        end, tab = parse_table(r.text)
        web[k] = tab
        ends.add(end)
        rep(k, "{}일 ({}~{})".format(len(tab), min(tab), max(tab)), "HTTP {}".format(r.status_code))
    if len(ends) != 1:
        sys.exit("[중단] 4쪽의 최종 거래일 불일치 {} — 재실행".format(sorted(ends)))
    days = sorted(set.intersection(*[set(v) for v in web.values()]))

    print("\n■ ② API 원문 대조 (웹 값 신뢰성)")
    n_cmp, bad = 0, []
    targets = []
    for d in days:
        api = load_api(d)
        if api is None or len(api) < EXPECT:
            targets.append(d)
            rep(d, "API 원문 {}".format("없음" if api is None else "{}행".format(len(api))), "보완 후보")
            continue
        for it in build_items(d, web):
            ref = api.get((it["areaName"], int(it["hour"])))
            if ref is None:
                continue
            for f, tol in (("smp", 0.005), ("mlfd", 0.5), ("jlfd", 0.5)):
                n_cmp += 1
                if ref.get(f) is None or abs(float(ref[f]) - it[f]) > tol:
                    bad.append("{} {}{}h {} api={} web={}".format(
                        d, it["areaName"], it["hour"], f, ref.get(f), it[f]))
    rep("대조 항목", "{:,}건".format(n_cmp), "불일치 {}건".format(len(bad)))
    for b in bad[:10]:
        print("      ! " + b)
    if bad:
        sys.exit("[중단] 웹·API 값 불일치 — 웹 구조 변경 가능성, 저장하지 않음")
    if n_cmp == 0:
        sys.exit("[중단] 대조 가능한 API 원문 없음 — 웹 값 검증 불가, 저장하지 않음")

    if a.dates:
        want = {x.strip() for x in a.dates.split(",") if x.strip()}
        targets = [d for d in targets if d in want]

    print("\n■ ③ 보완 저장")
    saved = []
    for d in targets:
        probs = valid(d, web)
        if probs:
            rep(d, "보류", " / ".join(probs))
            continue
        items = build_items(d, web)
        if a.dry_run:
            rep(d, "저장 예정 {}행".format(len(items)), "--dry-run")
            continue
        os.makedirs(RAW_WEB, exist_ok=True)
        js = {"response": {"header": {"resultCode": "00", "resultMsg": "KPX_WEB_SUPPLEMENT",
                                      "source": "kpx_web", "collectedAt": ts},
                           "body": {"dataType": "JSON", "totalCount": len(items),
                                    "items": {"item": items}}}}
        p = os.path.join(RAW_WEB, "{}.json".format(d))
        with open(p, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False)
        lm = sum(x["smp"] for x in items[:24]) / 24
        rep(d, "저장 {}행".format(len(items)), "육지 SMP 평균 {:.2f}원 → {}".format(lm, os.path.relpath(p, BASE_DIR)))
        saved.append(d)
    if not targets:
        rep("보완 대상", "없음", "API 원문 완비")

    os.makedirs(LOG_DIR, exist_ok=True)
    lp = os.path.join(LOG_DIR, "collect_kpx_web_{}.csv".format(ts))
    with open(lp, "w", encoding="utf-8-sig") as f:
        f.write("항목,값,비고\n")
        for r_ in _rows:
            f.write('"{}","{}","{}"\n'.format(r_["항목"], r_["값"], r_["비고"]))
    print("\n" + "=" * 78)
    print(" 결과 : 보완 저장 {}일{} / 로그 {}".format(
        len(saved), " ({})".format(", ".join(saved)) if saved else "", os.path.relpath(lp, BASE_DIR)))
    if saved:
        print(" 다음 : 전처리~피처 재생성 → step6_predict.py --backfill-only → 웹하우스 적재")
    print("=" * 78)


if __name__ == "__main__":
    main()
