# -*- coding: utf-8 -*-
"""
==============================================================================
 step12_kpx_sukub_poll.py — KPX 전력수급(5분) → Postgres 적재
==============================================================================
 원천(공공데이터포털 개발계정 승인 · _GW):
   · 오늘전력수급현황조회_GW  — apis.data.go.kr …/Sukub5mToday/getSukub5mToday
     (dataType=JSON, pageNo, numOfRows 필수)
   · 현재전력수급현황조회_GW  — openapi.kpx …/getSukub5mMaxDatetime (보조)
 인증 : config/.env 의 KPX_API_KEY / KPX_SUKUB_KEY (Decoding 키)
 보조 : API 실패 시 KPX 실시간 전력수급현황 공개 페이지
 주기 : crontab 5분마다 권장 (GW 호출은 SUKUB_GW_EVERY_MIN 분 주기, 기본 20분)
 버전 : v1.4 (2026-09-23) — 개발계정 일일 한도(≈100회) 대응: GW 호출 주기 제한·한도 초과 즉시 중단
       v1.3 (2026-09-22) — data.go.kr GW JSON 경로 우선
==============================================================================
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

import requests

try:
    import psycopg
except ImportError:
    psycopg = None

ROOT = Path(__file__).resolve().parents[1]
ENC = "utf-8-sig"
KST = ZoneInfo("Asia/Seoul")

# 포털 「오늘전력수급현황조회_GW」 — data.go.kr (승인 키로 동작 확인)
GW_TODAY = "https://apis.data.go.kr/B552115/Sukub5mToday/getSukub5mToday"
# 포털 「현재전력수급현황조회_GW」 — 기관 URL (최신 1건 보조)
URL_CURRENT = "https://openapi.kpx.or.kr/openapi/sukub5mMaxDatetime/getSukub5mMaxDatetime"
URL_TODAY = "https://openapi.kpx.or.kr/openapi/sukub5mToday/getSukub5mToday"
GW_CURRENT_CANDIDATES = (
    "https://apis.data.go.kr/B552115/sukub5mMaxDatetime/getSukub5mMaxDatetime",
    "https://apis.data.go.kr/B552115/Sukub5mMaxDatetime/getSukub5mMaxDatetime",
)

PAGE_URL = "https://kpx.or.kr/powerinfoSubmain.es?mid=a10404030000"
PAGE_UA = "Mozilla/5.0 (compatible; smp-dashboard/1.0)"
TIMEOUT = 30


class QuotaExceeded(RuntimeError):
    pass


def gw_due(now: datetime | None = None) -> bool:
    """개발계정 일일 한도 내로 GW 호출 — 5분 cron 중 N분 경계 회차만 호출."""
    every = int(os.getenv("SUKUB_GW_EVERY_MIN") or "20")
    if every <= 5:
        return True
    now = now or datetime.now(KST)
    return (now.hour * 60 + now.minute) % every < 5


def load_env() -> None:
    env_path = ROOT / "config" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding=ENC, errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def dsn() -> str:
    load_env()
    return (
        f"host={os.getenv('PGHOST', '127.0.0.1')} "
        f"port={os.getenv('PGPORT', '5432')} "
        f"dbname={os.getenv('PGDATABASE', 'smp')} "
        f"user={os.getenv('PGUSER', 'smp')} "
        f"password={os.getenv('PGPASSWORD', '')}"
    )


def _text(el: ET.Element | None, tag: str) -> str | None:
    if el is None:
        return None
    child = el.find(tag)
    if child is not None and child.text is not None:
        return child.text.strip()
    for c in el.iter():
        if c.tag.endswith(tag) and c.text:
            return c.text.strip()
    return None


def _f(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip().replace(",", "")
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _row(ts: datetime, **vals) -> dict:
    row = {
        "ts": ts,
        "supply_mw": None,
        "demand_mw": None,
        "total_demand_mw": None,
        "forecast_peak_mw": None,
        "supply_reserve_mw": None,
        "supply_reserve_pct": None,
        "oper_reserve_mw": None,
        "oper_reserve_pct": None,
    }
    row.update(vals)
    return row


def _item_to_row(item: dict) -> dict | None:
    base = item.get("baseDatetime")
    if base is None:
        return None
    base = str(base).strip()
    if len(base) < 12:
        return None
    ts = datetime.strptime(base[:14], "%Y%m%d%H%M%S").replace(tzinfo=KST)
    return _row(
        ts,
        supply_mw=_f(item.get("suppAbility")),
        demand_mw=_f(item.get("currPwrTot")),
        forecast_peak_mw=_f(item.get("forecastLoad")),
        supply_reserve_mw=_f(item.get("suppReservePwr")),
        supply_reserve_pct=_f(item.get("suppReserveRate")),
        oper_reserve_mw=_f(item.get("operReservePwr")),
        oper_reserve_pct=_f(item.get("operReserveRate")),
    )


def _raise_from_text(text: str) -> None:
    if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in text or re.search(
        r'"returnReasonCode"\s*:\s*"22"|<returnReasonCode>22<', text
    ):
        raise QuotaExceeded("일일 트래픽 초과(22)")
    if (
        "SERVICE KEY IS NOT REGISTERED" in text
        or "SERVICE_KEY_IS_NOT_REGISTERED" in text
        or "<resultCode>30</resultCode>" in text
        or '"resultCode" : "30"' in text
        or '"resultCode":"30"' in text
    ):
        raise PermissionError("KEY_NOT_REGISTERED")
    if "NO_OPENAPI_SERVICE_ERROR" in text or "<resultCode>12</resultCode>" in text:
        raise RuntimeError("NO_OPENAPI_SERVICE — 경로 불일치")
    # JSON header
    m_j = re.search(r'"resultCode"\s*:\s*"([^"]+)"', text)
    m_j2 = re.search(r'"resultMsg"\s*:\s*"([^"]+)"', text)
    if m_j and m_j.group(1) != "00":
        code = m_j.group(1)
        msg = m_j2.group(1) if m_j2 else text[:120]
        if code == "20":
            raise PermissionError(
                "서비스 접근거부(20) — 「오늘·현재전력수급현황조회_GW」 승인·반영을 확인하세요."
            )
        if code == "22":
            raise QuotaExceeded("일일 트래픽 초과(22)")
        if code == "11":
            raise RuntimeError(f"필수 파라미터 누락(11): {msg}")
        raise RuntimeError(f"API 오류 {code}: {msg}")
    if "<resultCode>" in text and "<resultCode>00</resultCode>" not in text:
        m = re.search(r"<resultCode>([^<]+)</resultCode>", text)
        m2 = re.search(r"<resultMsg>([^<]+)</resultMsg>", text)
        code = m.group(1) if m else "?"
        if code == "20":
            raise PermissionError(
                "서비스 접근거부(20) — 「오늘·현재전력수급현황조회_GW」 승인·반영을 확인하세요."
            )
        if code == "22":
            raise QuotaExceeded("일일 트래픽 초과(22)")
        raise RuntimeError(f"API 오류 {code}: {m2.group(1) if m2 else text[:120]}")


def _parse_payload(content: bytes, text: str) -> list[dict]:
    _raise_from_text(text)
    text_s = text.lstrip()
    rows: list[dict] = []
    if text_s.startswith("{") or text_s.startswith("["):
        data = json.loads(text)
        body = (data.get("response") or {}).get("body") or {}
        items = body.get("items") or body.get("item")
        item = None
        if isinstance(items, dict):
            item = items.get("item")
        elif isinstance(items, list):
            item = items
        elif items is not None:
            item = items
        if item is None and isinstance(body, dict) and "baseDatetime" in body:
            item = body
        if isinstance(item, dict):
            item = [item]
        if not isinstance(item, list):
            raise RuntimeError(f"JSON item 부재 — 응답 앞 200자: {text[:200]}")
        for it in item:
            if isinstance(it, dict):
                row = _item_to_row(it)
                if row:
                    rows.append(row)
    else:
        root = ET.fromstring(content)
        items = [el for el in root.iter() if el.tag.endswith("item")]
        if not items:
            items = [root]
        for el in items:
            it = {
                "baseDatetime": _text(el, "baseDatetime"),
                "suppAbility": _text(el, "suppAbility"),
                "currPwrTot": _text(el, "currPwrTot"),
                "forecastLoad": _text(el, "forecastLoad"),
                "suppReservePwr": _text(el, "suppReservePwr"),
                "suppReserveRate": _text(el, "suppReserveRate"),
                "operReservePwr": _text(el, "operReservePwr"),
                "operReserveRate": _text(el, "operReserveRate"),
            }
            row = _item_to_row(it)
            if row:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"baseDatetime 부재 — 응답 앞 200자: {text[:200]}")
    return rows


def _js_array(html: str, name: str) -> list[str]:
    m = re.search(rf"var\s+{name}\s*=\s*\[([^\]]*)\]", html)
    if not m:
        return []
    return [tok.strip().strip("'\"") for tok in m.group(1).split(",")]


def fetch_page() -> list[dict]:
    """보조 경로 — KPX 실시간 전력수급현황 공개 페이지(인증키 불요)."""
    r = requests.get(PAGE_URL, headers={"User-Agent": PAGE_UA}, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text

    times = _js_array(html, "t_time")
    market = _js_array(html, "x")
    total = _js_array(html, "v")
    rows: dict[datetime, dict] = {}
    for i, t in enumerate(times):
        if not re.fullmatch(r"\d{14}", t):
            continue
        demand = _f(market[i]) if i < len(market) else None
        tot = _f(total[i]) if i < len(total) else None
        if demand is None and tot is None:
            continue
        ts = datetime.strptime(t, "%Y%m%d%H%M%S").replace(tzinfo=KST)
        rows[ts] = _row(ts, demand_mw=demand, total_demand_mw=tot)

    text = re.sub(r"(?is)<script.*?</script>", " ", html)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))

    def grab(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return _f(m.group(1)) if m else None

    snap = {
        "supply_mw": grab(r"공급능력\s*([\d,]+)\s*MW"),
        "demand_mw": grab(r"현재수요\(전력시장\)\s*([\d,]+)\s*MW"),
        "supply_reserve_mw": grab(r"공급예비력\s*([\d,]+)\s*MW"),
        "supply_reserve_pct": grab(r"공급예비율\s*([\d.]+)\s*%"),
    }
    m = re.search(r"(20\d{2})\.(\d{2})\.(\d{2})\([^)]*\)\s*(\d{2}):(\d{2})", text)
    if m and any(v is not None for v in snap.values()):
        y, mo, d, hh, mi = (int(g) for g in m.groups())
        ts = datetime(y, mo, d, hh, mi, tzinfo=KST)
        base = rows.get(ts) or _row(ts)
        for k, v in snap.items():
            if v is not None:
                base[k] = v
        rows[ts] = base

    if not rows:
        raise RuntimeError("KPX 실시간 페이지 파싱 실패 — 화면 구조 변경 여부 확인")
    return [rows[k] for k in sorted(rows)]


def _api_keys() -> list[str]:
    load_env()
    keys: list[str] = []
    for name in ("KPX_SUKUB_KEY", "KPX_API_KEY"):
        v = (os.getenv(name) or "").strip()
        if v and v not in keys:
            keys.append(v)
    if not keys:
        raise RuntimeError("KPX_API_KEY(또는 KPX_SUKUB_KEY) 미설정 — config/.env 확인")
    return keys


def _get_gw_today(key: str) -> requests.Response:
    """오늘수급_GW — Decoding 키 + JSON + 페이징(필수)."""
    return requests.get(
        GW_TODAY,
        params={
            "serviceKey": unquote(key),
            "pageNo": "1",
            "numOfRows": "400",
            "dataType": "JSON",
        },
        timeout=TIMEOUT,
    )


def _get_generic(url: str, key: str, mode: str, extra: dict | None = None) -> requests.Response:
    params = dict(extra or {})
    if mode == "params_decoded":
        params["serviceKey"] = unquote(key)
        return requests.get(url, params=params, timeout=TIMEOUT)
    if mode == "params_decoded_ServiceKey":
        params["ServiceKey"] = unquote(key)
        return requests.get(url, params=params, timeout=TIMEOUT)
    if mode == "raw_url_ServiceKey":
        q = "&".join(f"{k}={v}" for k, v in params.items())
        suffix = f"&{q}" if q else ""
        return requests.get(f"{url}?ServiceKey={key}{suffix}", timeout=TIMEOUT)
    if mode == "raw_url_enc_ServiceKey":
        q = "&".join(f"{k}={v}" for k, v in params.items())
        suffix = f"&{q}" if q else ""
        return requests.get(
            f"{url}?ServiceKey={quote(unquote(key), safe='')}{suffix}",
            timeout=TIMEOUT,
        )
    raise ValueError(mode)


def fetch() -> list[dict]:
    """오늘·현재 수급 OpenAPI — _GW 승인 경로 우선."""
    keys = _api_keys()
    NOT_REGISTERED = (
        "수급 API에 키가 미등록입니다. 공공데이터포털 "
        "「오늘전력수급현황조회_GW」(15056879)·「현재전력수급현황조회_GW」(15056640) "
        "승인 상태와 반영을 확인하세요."
    )
    errors: list[str] = []

    for key in keys:
        # 1) 오늘수급_GW (data.go.kr) — 최우선
        try:
            r = _get_gw_today(key)
            return _parse_payload(r.content, r.text)
        except QuotaExceeded:
            raise
        except Exception as e:
            errors.append(f"GW_TODAY: {e}")

        # 2) 기관 URL · 현재/오늘 (반영 지연 시 대비)
        page_extra = {"pageNo": "1", "numOfRows": "400", "dataType": "JSON"}
        for url, mode, extra in (
            (URL_TODAY, "raw_url_enc_ServiceKey", None),
            (URL_CURRENT, "raw_url_enc_ServiceKey", None),
            (URL_TODAY, "params_decoded", page_extra),
            (URL_CURRENT, "params_decoded", {"pageNo": "1", "numOfRows": "10", "dataType": "JSON"}),
            *((u, "params_decoded", {"pageNo": "1", "numOfRows": "10", "dataType": "JSON"})
              for u in GW_CURRENT_CANDIDATES),
        ):
            if mode.startswith("raw_url") and "%" not in key and "enc" not in mode:
                continue
            try:
                r = _get_generic(url, key, mode, extra)
                if r.status_code >= 400 and "resultCode" not in r.text and "_ERROR" not in r.text:
                    r.raise_for_status()
                return _parse_payload(r.content, r.text)
            except PermissionError as e:
                errors.append(NOT_REGISTERED if str(e) == "KEY_NOT_REGISTERED" else str(e))
            except Exception as e:
                errors.append(f"{url.split('/')[-1]}|{mode}: {e}")

    for msg in errors:
        if msg != NOT_REGISTERED and "NO_OPENAPI_SERVICE" not in msg:
            raise RuntimeError(msg)
    raise RuntimeError(errors[0] if errors else "수급 API 호출 실패")


def upsert_many(rows: list[dict]) -> int:
    if psycopg is None:
        raise RuntimeError("psycopg 미설치")
    sql = """
    INSERT INTO smp.fact_power_balance (
        ts, supply_mw, demand_mw, total_demand_mw, forecast_peak_mw,
        supply_reserve_mw, supply_reserve_pct,
        oper_reserve_mw, oper_reserve_pct
    ) VALUES (
        %(ts)s, %(supply_mw)s, %(demand_mw)s, %(total_demand_mw)s, %(forecast_peak_mw)s,
        %(supply_reserve_mw)s, %(supply_reserve_pct)s,
        %(oper_reserve_mw)s, %(oper_reserve_pct)s
    )
    ON CONFLICT (ts) DO UPDATE SET
        supply_mw=COALESCE(EXCLUDED.supply_mw, smp.fact_power_balance.supply_mw),
        demand_mw=COALESCE(EXCLUDED.demand_mw, smp.fact_power_balance.demand_mw),
        total_demand_mw=COALESCE(EXCLUDED.total_demand_mw, smp.fact_power_balance.total_demand_mw),
        forecast_peak_mw=COALESCE(EXCLUDED.forecast_peak_mw, smp.fact_power_balance.forecast_peak_mw),
        supply_reserve_mw=COALESCE(EXCLUDED.supply_reserve_mw, smp.fact_power_balance.supply_reserve_mw),
        supply_reserve_pct=COALESCE(EXCLUDED.supply_reserve_pct, smp.fact_power_balance.supply_reserve_pct),
        oper_reserve_mw=COALESCE(EXCLUDED.oper_reserve_mw, smp.fact_power_balance.oper_reserve_mw),
        oper_reserve_pct=COALESCE(EXCLUDED.oper_reserve_pct, smp.fact_power_balance.oper_reserve_pct),
        loaded_at=now()
    """
    with psycopg.connect(dsn()) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(sql, row)
        conn.commit()
    return len(rows)


def collect() -> tuple[list[dict], list[str]]:
    """OpenAPI 우선, 실패 시 페이지 보조. 둘 다 있으면 병합."""
    merged: dict[datetime, dict] = {}
    sources: list[str] = []
    errors: list[str] = []

    if gw_due():
        try:
            api_rows = fetch()
            for r in api_rows:
                merged[r["ts"]] = r
            sources.append("OpenAPI")
        except Exception as e:
            errors.append(f"OpenAPI: {e}")

    try:
        for r in fetch_page():
            base = merged.get(r["ts"]) or _row(r["ts"])
            for k, v in r.items():
                if k != "ts" and v is not None and base.get(k) is None:
                    base[k] = v
            # 페이지 total_demand 는 OpenAPI에 없으므로 항상 보강
            if r.get("total_demand_mw") is not None:
                base["total_demand_mw"] = r["total_demand_mw"]
            merged[r["ts"]] = base
        sources.append("실시간 페이지")
    except Exception as e:
        errors.append(f"실시간 페이지: {e}")

    if not merged:
        raise RuntimeError(" / ".join(errors) or "수집 경로 없음")
    if errors:
        print(f"[안내] 일부 경로 미사용 — {' / '.join(errors)}")
    return [merged[k] for k in sorted(merged)], sources


def main() -> int:
    try:
        rows, sources = collect()
        source = "+".join(sources)
        n = upsert_many(rows)
        latest = max(rows, key=lambda r: r["ts"])
        print(
            f"[ok/{source}] {n}건 적재 · 최신 {latest['ts']:%Y-%m-%d %H:%M} "
            f"수요={latest['demand_mw']} 공급={latest['supply_mw']} "
            f"예비율={latest['supply_reserve_pct']}%"
        )
        return 0
    except Exception as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
