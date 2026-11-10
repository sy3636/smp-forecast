# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 보조점검(SMP 확정 공표 시점 실측)
================================================
배경 : SMP는 하루전시장에서 D-1에 결정되며, 공공데이터포털 API가 D-1 저녁부터
       D일 24시간 확정값을 제공함이 확인됨(2026.8.11 — 8/12분이 8/11 22:09 수집)
       → 예측 가치가 발생하는 구간을 정하려면 **제공 개시 시각**의 실측이 필요

목적 : 현재 시각 기준으로 D+0 ~ D+3 각 일자의 SMP·수요예측 제공 여부를 조회하여
       ① 어느 지평까지 확정값이 공개되는지 ② 공개 개시 시각이 언제인지 규명

사용법 : 하루 중 여러 시각(예: 09시·12시·15시·18시·21시)에 반복 실행하여
        결과를 누적하면 제공 개시 시각의 경계를 좁힐 수 있음

  python scripts\\step1_probe_publish.py            (D+0~D+3 조회)
  python scripts\\step1_probe_publish.py --ahead 5  (D+0~D+5)

판정
  · 확정 제공 : 24시간 전량 조회됨 → 해당 지평은 예측 대상이 아닌 **조회 대상**
  · 미제공    : 0건 → 예측 가치 있음

주의 : 개발계정 트래픽 100건/일 → --ahead 값 과다 설정 금지
출력 : logs/probe_publish_{일시}.csv  (누적 분석용)

버전 : v1.0 (2026-08-12)
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import (  # noqa: E402
    BASE_DIR, EP_PATH, ENV_PATH, load_env, parse_result, count_items, TIMEOUT,
)

try:
    import requests
except ImportError:
    sys.exit("[중단] requests 미설치 → pip install -r requirements.txt")

LOG_DIR = os.path.join(BASE_DIR, "logs")
HIST = os.path.join(LOG_DIR, "probe_publish_history.csv")


def to_int(v):
    """건수 문자열을 정수로 변환 — 미상 값은 0 처리"""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def find_api():
    """계통한계가격 엔드포인트 정의 조회"""
    with open(EP_PATH, encoding="utf-8") as f:
        eps = json.load(f)
    items = eps.get("apis", eps) if isinstance(eps, dict) else eps
    for e in items:
        if isinstance(e, dict) and "계통한계가격" in str(e.get("name", "")):
            return e
    return None


def count_demand(text):
    """응답에서 수요예측 값(mlfd·jlfd)이 채워진 건수 계수

    비고 : SMP는 공표되었으나 수요예측이 비어 있는 시간대가 존재하므로
           (2026.8.12 실측) 예측 시각 투입 가능 여부는 본 계수로 판정함
    """
    n = 0
    for tag in ("mlfd", "jlfd"):
        for m in re.finditer(rf"<{tag}>([^<]*)</{tag}>", text):
            v = m.group(1).strip()
            if v and v not in ("0", "0.0"):
                n += 1
                break_flag = True
        if n:
            break                      # 육지(mlfd) 우선, 없으면 제주(jlfd)
    if n == 0:                          # JSON 응답 대비
        for key in ("mlfd", "jlfd"):
            n = max(n, len([1 for m in re.finditer(rf'"{key}"\s*:\s*"?([0-9.]+)"?', text)
                            if float(m.group(1) or 0) > 0]))
            if n:
                break
    return n


def main():
    ap = argparse.ArgumentParser(description="SMP 확정 공표 시점 실측")
    ap.add_argument("--ahead", type=int, default=3,
                    help="조회할 미래 지평(일). 기본 3 — 트래픽 유의")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now()
    ts = f"{now:%Y%m%d_%H%M%S}"

    print("=" * 74)
    print(" step1 SMP 확정 공표 시점 실측 (v1.0)")
    print(f" 실행시각 : {now:%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    api = find_api()
    if not api:
        sys.exit("[중단] 계통한계가격 엔드포인트 정의를 찾을 수 없음 → config/endpoints.json")
    env = load_env(ENV_PATH)
    key = env.get(api.get("key_env", "KPX_API_KEY"), "")
    if not key:
        sys.exit("[중단] 인증키 없음 → config/.env 확인")

    print(f"\n  {'지평':<6}{'일자':<12}{'HTTP':>6}{'결과':>6}{'SMP':>6}{'수요예측':>8}   판정")
    print("  " + "-" * 68)

    rows = []
    for k in range(0, args.ahead + 1):
        d = now + timedelta(days=k)
        ymd = f"{d:%Y%m%d}"
        params = {kk: vv for kk, vv in api.get("params", {}).items()
                  if not str(vv).startswith("{")}
        params["date"] = ymd
        params[api.get("key_param", "serviceKey")] = key
        try:
            r = requests.get(api["url"], params=params, timeout=TIMEOUT)
            code, msg = parse_result(r.text)
            cnt = to_int(count_items(r.text))    # 문자열 반환 대비 정수 변환
            http = r.status_code
        except Exception as e:
            code, msg, cnt, http = "-", str(e)[:30], 0, "-"

        # 수요예측 항목(mlfd·jlfd) 별도 계수 — SMP와 공표 시점이 다를 수 있음
        #  · 2026.8.12 실측 : 17시에 SMP 48건 / 수요예측 0건 확인
        #  · 예측 시각(14~15시) 투입 가능 여부는 **수요예측 건수**로 판정해야 함
        try:
            dm_cnt = count_demand(r.text)
        except Exception:
            dm_cnt = 0

        state = "확정 제공" if cnt >= 24 else ("일부" if cnt else "미제공")
        if cnt >= 24 and dm_cnt >= 24:
            note = "SMP·수요예측 모두 공개"
        elif cnt >= 24:
            note = "SMP만 공개 — 수요예측 미공개"
        elif dm_cnt >= 24:
            note = "수요예측만 공개 — **투입 가능**"
        else:
            note = "예측 가치 있음"
        print(f"  D+{k:<4}{ymd:<12}{str(http):>6}{str(code):>6}{cnt:>6}{dm_cnt:>8}   {state} ({note})")
        rows.append({"실행시각": f"{now:%Y-%m-%d %H:%M:%S}", "지평": f"D+{k}",
                     "일자": ymd, "HTTP": http, "결과코드": code,
                     "건수": cnt, "수요예측건수": dm_cnt, "판정": state})

    print("  " + "-" * 60)

    conf = [r for r in rows if r["건수"] >= 24]
    last = max((int(r["지평"][2:]) for r in conf), default=-1)
    print(f"\n  · 확정 공개 지평 : D+{last}" if last >= 0 else "\n  · 확정 공개 지평 : 없음")
    print(f"  · 예측 가치 구간 : D+{last + 1} 이상")

    # 누적 이력 저장 — 여러 시각 실행분을 모아 공개 개시 시각 추정
    new = not os.path.exists(HIST)
    with open(HIST, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"  → 누적 이력 {HIST}")

    print("=" * 74)
    print(" 활용 : 하루 중 여러 시각에 반복 실행하여 공개 개시 시각의 경계를 확인")
    print("=" * 74)


if __name__ == "__main__":
    main()
