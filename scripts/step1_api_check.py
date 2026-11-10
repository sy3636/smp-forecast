# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 구축 - step1 사전점검
================================================
목적 : ① config/.env 인증키 유효성 검증
       ② config/endpoints.json 에 등록된 API 엔드포인트 연결 테스트
       ③ 점검 결과를 콘솔 표 + CSV 로그로 출력
버전 : v1.1 (2026-07-29) — 실제 엔드포인트 반영, 날짜토큰 치환·키변수 동적 인식 추가
실행 : python step1_api_check.py            (실제 호출)
       python step1_api_check.py --dry-run  (네트워크 미사용, 설정만 점검)
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    sys.exit("[중단] requests 미설치 → 실행: pip install requests python-dotenv")

# ------------------------------------------------------------------
# 0. 경로 설정
# ------------------------------------------------------------------
def find_root(start):
    """스크립트 위치에서 상위로 탐색하여 config 폴더 보유 경로를 프로젝트 루트로 인식."""
    cur = start
    for _ in range(4):
        if os.path.isdir(os.path.join(cur, "config")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return start


BASE_DIR = find_root(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, "config", ".env")
EP_PATH = os.path.join(BASE_DIR, "config", "endpoints.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TIMEOUT = 30  # 초 (포털 개편 후 응답 지연 대응)

# 공공데이터포털 공통 오류코드 해설
ERR_GUIDE = {
    "00": "정상",
    "01": "어플리케이션 에러(제공기관 서버 오류)",
    "02": "데이터베이스 에러",
    "03": "데이터 없음(파라미터 조건 재확인)",
    "04": "HTTP 에러",
    "05": "서비스 연결 실패",
    "10": "잘못된 요청 파라미터",
    "11": "필수 요청 파라미터 누락",
    "12": "해당 오픈API 서비스 폐기",
    "20": "서비스 접근 거부(활용신청 미승인)",
    "22": "서비스 요청 제한횟수 초과",
    "30": "등록되지 않은 인증키(키 오류 또는 승인 전)",
    "31": "기한 만료된 인증키",
    "32": "등록되지 않은 도메인/IP",
    "99": "기타 오류",
}


# ------------------------------------------------------------------
# 1. .env 로딩 (python-dotenv 없이도 동작하도록 자체 파서 병행)
# ------------------------------------------------------------------
def load_env(path):
    """.env 파일을 읽어 dict 반환. 주석·공백·따옴표 처리 포함."""
    if not os.path.exists(path):
        # .env.txt 오저장 여부 안내
        alt = path + ".txt"
        msg = f"[중단] .env 파일 없음 → {path}"
        if os.path.exists(alt):
            msg += "\n       ※ '.env.txt' 발견. 메모장 저장 시 파일형식을 '모든 파일'로 지정 후 '.env'로 재저장 필요"
        sys.exit(msg)

    env = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

    # ── 인증키 자동 정규화 ────────────────────────
    # 공공데이터포털은 Encoding·Decoding 두 형태를 제공하나,
    # requests 의 params 방식은 Decoding 키를 요구함(Encoding 키 사용 시 이중 인코딩 → 403).
    # 포털 개편 후 Decoding 키가 표시되지 않는 경우가 있어, Encoding 키를 자동 변환함.
    for k in [k for k in env if k.endswith("_API_KEY")]:
        v = env[k]
        if v and "%" in v:
            dec = urllib.parse.unquote(v)
            if dec != v:
                env[k] = dec
                env.setdefault("_AUTO_DECODED", "")
                env["_AUTO_DECODED"] += f"{k} "
    return env


def diagnose_key(name, value):
    """인증키 형식 사전 진단. (상태, 메시지) 반환."""
    if not value:
        return "오류", "미입력"
    if value.startswith("발급받은") or "여기에" in value:
        return "오류", "예시값 그대로 → 실제 키로 교체 필요"
    if value != value.strip():
        return "경고", "앞뒤 공백 포함 → 제거 필요"
    if "%" in value:
        return "경고", "Encoding 키 잔존 → 자동 변환 실패, 포털에서 키 재확인 필요"
    if len(value) < 40:
        return "경고", f"길이 {len(value)}자 (일반적으로 80자 내외) → 복사 누락 확인"
    return "정상", f"{len(value)}자 / {value[:6]}…{value[-4:]}"


# ------------------------------------------------------------------
# 2. 응답 판정
# ------------------------------------------------------------------
def parse_result(text):
    """JSON·XML 응답에서 resultCode / resultMsg 추출."""
    # (1) JSON 시도
    try:
        js = json.loads(text)
        hdr = js.get("response", {}).get("header", js.get("header", {}))
        code = str(hdr.get("resultCode", hdr.get("returnReasonCode", ""))).zfill(2)
        msg = hdr.get("resultMsg", hdr.get("returnAuthMsg", ""))
        if code:
            return code, msg
    except Exception:
        pass

    # (2) XML / 텍스트 시도
    pat = {
        "code": r"<(?:resultCode|returnReasonCode)>\s*([^<]+?)\s*</",
        "msg": r"<(?:resultMsg|returnAuthMsg|errMsg)>\s*([^<]+?)\s*</",
    }
    c = re.search(pat["code"], text)
    m = re.search(pat["msg"], text)
    code = c.group(1).strip().zfill(2) if c else ""
    msg = m.group(1).strip() if m else ""

    # (3) 인증키 오류 특유 문구 탐지
    if not code and "SERVICE_KEY_IS_NOT_REGISTERED" in text:
        return "30", "SERVICE_KEY_IS_NOT_REGISTERED_ERROR"
    return code, msg


def count_items(text):
    """응답 내 데이터 건수 추정."""
    try:
        js = json.loads(text)
        body = js.get("response", {}).get("body", js.get("body", {}))
        if "totalCount" in body:
            return str(body["totalCount"])
        items = body.get("items", {})
        if isinstance(items, dict):
            items = items.get("item", [])
        if isinstance(items, list):
            return str(len(items))
    except Exception:
        pass
    m = re.search(r"<totalCount>\s*(\d+)\s*</totalCount>", text)
    if m:
        return m.group(1)
    return str(text.count("<item>")) if "<item>" in text else "-"


# ------------------------------------------------------------------
# 3. 개별 API 점검
# ------------------------------------------------------------------
def subst_tokens(params):
    """params 값의 날짜 토큰을 실행시점 기준으로 치환."""
    now = datetime.now()
    table = {
        "{TODAY}": f"{now:%Y%m%d}",
        "{YESTERDAY}": f"{now - timedelta(days=1):%Y%m%d}",
        "{TOMORROW}": f"{now + timedelta(days=1):%Y%m%d}",
        "{D-3}": f"{now - timedelta(days=3):%Y%m%d}",   # 실적자료 제공 지연 대비
        "{D-7}": f"{now - timedelta(days=7):%Y%m%d}",
        "{YEAR}": f"{now:%Y}",
        "{MONTH}": f"{now:%m}",
    }
    out = {}
    for k, v in params.items():
        v = str(v)
        for tok, val in table.items():
            v = v.replace(tok, val)
        out[k] = v
    return out


def check_api(api, env, dry_run=False):
    name = api.get("name", "(무명)")
    url = api.get("url", "")
    key_env = api.get("key_env", "")
    key_param = api.get("key_param", "serviceKey")
    params = subst_tokens(api.get("params", {}))
    key = env.get(key_env, "")

    row = {
        "API명": name,
        "키변수": key_env,
        "HTTP": "-",
        "결과코드": "-",
        "건수": "-",
        "판정": "",
        "비고": "",
    }

    # 설정 사전 점검
    if not url or url.startswith("<"):
        row["판정"] = "미설정"
        row["비고"] = "endpoints.json 의 url 을 활용가이드 요청주소로 교체 필요"
        return row
    if not key:
        row["판정"] = "실패"
        row["비고"] = f".env 에 {key_env} 없음"
        return row
    if dry_run:
        row["판정"] = "점검생략"
        row["비고"] = "dry-run 모드(설정 정상)"
        return row

    params[key_param] = key
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        row["HTTP"] = str(r.status_code)
        text = r.text or ""
        code, msg = parse_result(text)
        row["결과코드"] = code or "-"
        row["건수"] = count_items(text)

        if r.status_code != 200:
            row["판정"] = "실패"
            row["비고"] = f"HTTP {r.status_code}"
        elif code in ("00", "0", ""):
            row["판정"] = "정상"
            row["비고"] = msg or "OK"
        else:
            row["판정"] = "실패"
            row["비고"] = f"{ERR_GUIDE.get(code, msg or '알 수 없는 코드')}"
    except requests.exceptions.Timeout:
        row["판정"] = "실패"
        row["비고"] = f"타임아웃({TIMEOUT}초) → 사내망 프록시 여부 확인"
    except requests.exceptions.SSLError:
        row["판정"] = "실패"
        row["비고"] = "SSL 오류 → http/https 확인 또는 사내 인증서 설정 필요"
    except Exception as e:
        row["판정"] = "실패"
        row["비고"] = f"{type(e).__name__}: {str(e)[:60]}"
    return row


# ------------------------------------------------------------------
# 4. 출력 유틸
# ------------------------------------------------------------------
def w(s):
    """한글 폭 보정용 표시길이 계산."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))


def pad(s, n):
    return str(s) + " " * max(0, n - w(s))


def print_table(rows, cols):
    widths = {c: max(w(c), max((w(r[c]) for r in rows), default=0)) for c in cols}
    line = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    print(line)
    print("| " + " | ".join(pad(c, widths[c]) for c in cols) + " |")
    print(line)
    for r in rows:
        print("| " + " | ".join(pad(r[c], widths[c]) for c in cols) + " |")
    print(line)


# ------------------------------------------------------------------
# 5. 메인
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step1 API 사전점검")
    ap.add_argument("--dry-run", action="store_true", help="네트워크 호출 없이 설정만 점검")
    args = ap.parse_args()

    print("=" * 70)
    print(" SMP 예측 프로그램 - step1 API 사전점검 (v1.1)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # (0) 설정파일 로딩
    if not os.path.exists(EP_PATH):
        sys.exit(f"[중단] 설정파일 없음 → {EP_PATH}")
    with open(EP_PATH, "r", encoding="utf-8") as f:
        conf = json.load(f)
    apis = conf.get("apis", [])

    # (1) 인증키 점검
    env = load_env(ENV_PATH)
    key_envs = []
    for a in apis:
        ke = a.get("key_env", "")
        if ke and ke not in key_envs:
            key_envs.append(ke)

    # 미기재 키는 최초 유효키로 대체(공공데이터포털은 계정당 단일 인증키)
    primary = next((env[k] for k in key_envs if env.get(k)), "")
    if env.get("_AUTO_DECODED"):
        print(f"  ※ Encoding 키 자동 변환 적용 : {env['_AUTO_DECODED'].strip()}")
    for k in key_envs:
        if not env.get(k) and primary:
            env[k] = primary
            print(f"  ※ {k} 미기재 → 계정 공통키로 자동 대체")

    print("\nⅠ. 인증키 점검")
    key_rows = []
    for k in key_envs:
        st, msg = diagnose_key(k, env.get(k, ""))
        key_rows.append({"환경변수": k, "상태": st, "내용": msg})
    print_table(key_rows, ["환경변수", "상태", "내용"])

    vals = {env.get(k, "") for k in key_envs if env.get(k)}
    if len(key_envs) > 1 and len(vals) == 1:
        print("  ※ 전 키 동일 → data.go.kr 계정 단위 발급키 사용 형태로 정상")

    if any(r["상태"] == "오류" for r in key_rows):
        sys.exit("\n[중단] 인증키 오류 → config/.env 수정 후 재실행")

    print(f"\nⅡ. 엔드포인트 연결 점검 (대상 {len(apis)}건"
          + (", dry-run" if args.dry_run else "") + ")")
    rows = [check_api(a, env, args.dry_run) for a in apis]
    cols = ["API명", "키변수", "HTTP", "결과코드", "건수", "판정", "비고"]
    print_table(rows, cols)

    # (3) 요약
    ok = sum(1 for r in rows if r["판정"] == "정상")
    ng = sum(1 for r in rows if r["판정"] == "실패")
    na = len(rows) - ok - ng
    print(f"\nⅢ. 요약 : 정상 {ok}건 / 실패 {ng}건 / 미설정·생략 {na}건")

    # (4) 로그 저장
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"step1_api_check_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"    로그 저장 : {log_path}")

    if ng == 0 and na == 0:
        print("\n[완료] 전 API 정상 → step1(데이터 수집) 진행 가능")
    else:
        print("\n[조치필요] 비고란 확인 후 재실행")
    sys.exit(0 if ng == 0 else 1)


if __name__ == "__main__":
    main()
