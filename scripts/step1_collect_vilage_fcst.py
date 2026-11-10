# -*- coding: utf-8 -*-
"""
step1_collect_vilage_fcst.py — 기상청 단기예보(동네예보) 수집·권역 집약
==============================================================================
목적
  · 예측 시점에 확보 가능한 익일 강수·하늘상태·기온을 운영 피처로 투입
  · ASOS는 전날까지만 개방되므로 D+1 기상은 본 API가 유일한 경로

입력 : config/endpoints.json (KMA_단기예보조회), config/.env
출력 : data/interim/kma_vilage_hourly.csv
       (date, hour_n, temp_land, humid_land, wind_spd_land, rain_land,
        sky_land, pty_land, pop_land, solar_land)

처리
  · 격자 : ASOS 육지 7지점 대응 nx·ny, 전력수요 비중 가중평균
  · PCP 문자열 파싱, SKY→일사 대리(월·시간 기후값 × 하늘상태 계수)
  · 기존 weather_hourly 의 결측 시각을 예보값으로 보완(원본 ASOS는 유지)

버전 : v1.0 (2026-09-01)
실행 : python scripts\\step1_collect_vilage_fcst.py
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR, ENV_PATH, TIMEOUT, load_env  # noqa: E402
from step1_collect import load_api, resolve_key  # noqa: E402
from step1_collect_weather import STATIONS, LAND_IDS  # noqa: E402

try:
    import numpy as np
    import pandas as pd
    import requests
except ImportError:
    sys.exit("[중단] pandas·numpy·requests 미설치")

INTERIM = os.path.join(BASE_DIR, "data", "interim")
WSRC = os.path.join(INTERIM, "weather_hourly.csv")
OUT = os.path.join(INTERIM, "kma_vilage_hourly.csv")

# 육지 ASOS 지점 → 동네예보 격자(nx, ny) 근사
GRID = {
    "108": (60, 127), "112": (55, 124), "133": (67, 100),
    "143": (89, 90), "156": (58, 74), "159": (98, 76),
    "105": (92, 131), "184": (53, 38),
}
SKY_SOLAR = {1: 1.00, 3: 0.45, 4: 0.18}   # 맑음·구름많음·흐림


def parse_pcp(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    s = str(v).strip()
    if s in ("", "강수없음", "0", "0.0"):
        return 0.0
    if "미만" in s:
        return 0.5
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else 0.0


def base_slot(now=None):
    """가장 최근 발표 시각(0200/0500/0800/1100/1400/1700/2000/2300)."""
    now = now or datetime.now()
    slots = [2, 5, 8, 11, 14, 17, 20, 23]
    h = now.hour
    m = now.minute
    # 발표 후 약 10분 지연을 감안
    avail = h if m >= 10 else h - 1
    chosen = max([s for s in slots if s <= avail], default=None)
    if chosen is None:
        d = now - timedelta(days=1)
        return f"{d:%Y%m%d}", "2300"
    return f"{now:%Y%m%d}", f"{chosen:02d}00"


def fetch_grid(url, key, base_date, base_time, nx, ny):
    p = {"serviceKey": key, "pageNo": "1", "numOfRows": "1000",
         "dataType": "JSON", "base_date": base_date, "base_time": base_time,
         "nx": str(nx), "ny": str(ny)}
    r = requests.get(url, params=p, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    items = (((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    return items


def solar_climo():
    """월×시간 일사 기후값 — ASOS 실측 중앙값."""
    if not os.path.exists(WSRC):
        return {}
    w = pd.read_csv(WSRC, encoding="utf-8-sig", usecols=["date", "hour_n", "solar_land"])
    w["month"] = pd.to_datetime(w["date"]).dt.month
    g = w.groupby(["month", "hour_n"])["solar_land"].median()
    return {(int(m), int(h)): float(v) for (m, h), v in g.items()}


def main():
    os.makedirs(INTERIM, exist_ok=True)
    env = load_env(ENV_PATH)
    api = load_api("단기예보")
    key = resolve_key(api, env)
    if not key:
        sys.exit("[중단] KMA_API_KEY 없음")
    bd, bt = base_slot()
    print("=" * 72)
    print(" step1 단기예보 수집 (v1.0)")
    print(f" 발표 : {bd} {bt}")
    print("=" * 72)

    recs = []
    for stn, (nx, ny) in GRID.items():
        try:
            items = fetch_grid(api["url"], key, bd, bt, nx, ny)
        except Exception as e:
            print(f"  · {stn} 실패 {type(e).__name__}: {e}")
            continue
        print(f"  · {STATIONS.get(stn, {}).get('name', stn)}({stn}) {len(items)}건")
        for it in items:
            recs.append({
                "stnId": stn,
                "fcstDate": str(it.get("fcstDate", "")),
                "fcstTime": str(it.get("fcstTime", "")),
                "category": it.get("category"),
                "value": it.get("fcstValue"),
                "weight": STATIONS.get(stn, {}).get("weight", 0) if stn != "184" else 0,
            })
    if not recs:
        sys.exit("[중단] 단기예보 수신 없음")

    raw = pd.DataFrame(recs)
    raw["date"] = pd.to_datetime(raw["fcstDate"], format="%Y%m%d", errors="coerce")
    raw["hour_n"] = (pd.to_numeric(raw["fcstTime"], errors="coerce") // 100).astype("Int64") + 1
    raw = raw.dropna(subset=["date", "hour_n"])
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    raw["hour_n"] = raw["hour_n"].clip(1, 24).astype(int)

    land = raw[raw["stnId"].isin(LAND_IDS)].copy()

    def wavg(sub, cat, mapper=None):
        s = sub[sub["category"] == cat].copy()
        if s.empty:
            return pd.DataFrame(columns=["date", "hour_n", cat.lower()])
        s["v"] = s["value"].map(mapper) if mapper else pd.to_numeric(s["value"], errors="coerce")
        s = s.dropna(subset=["v"])
        g = s.groupby(["date", "hour_n"], sort=False)
        out = g.apply(lambda x: np.average(x["v"], weights=x["weight"]) if x["weight"].sum() else x["v"].mean(),
                      include_groups=False)
        return out.rename(cat.lower()).reset_index()

    tmp = wavg(land, "TMP")
    reh = wavg(land, "REH")
    wsd = wavg(land, "WSD")
    pcp = wavg(land, "PCP", parse_pcp)
    sky = wavg(land, "SKY")
    pty = wavg(land, "PTY")
    pop = wavg(land, "POP")
    out = tmp.rename(columns={"tmp": "temp_land"})
    for d, c in ((reh, "reh"), (wsd, "wsd"), (pcp, "pcp"),
                 (sky, "sky"), (pty, "pty"), (pop, "pop")):
        if d.empty:
            continue
        nn = {"reh": "humid_land", "wsd": "wind_spd_land", "pcp": "rain_land",
              "sky": "sky_land", "pty": "pty_land", "pop": "pop_land"}[c]
        out = out.merge(d.rename(columns={c: nn}), on=["date", "hour_n"], how="outer")

    # PTY>0 인데 PCP가 0이면 약한 강수로 보정
    if "pty_land" in out.columns and "rain_land" in out.columns:
        out["rain_land"] = np.where(
            (out["rain_land"].fillna(0) <= 0) & (out["pty_land"].fillna(0) >= 1),
            0.8, out["rain_land"].fillna(0))

    climo = solar_climo()
    def sol_row(r):
        m = int(str(r["date"])[5:7])
        base = climo.get((m, int(r["hour_n"])), 0.0)
        sky = r.get("sky_land")
        k = SKY_SOLAR.get(int(round(sky)) if pd.notna(sky) else 3, 0.45)
        if pd.notna(r.get("pty_land")) and r["pty_land"] >= 1:
            k *= 0.35
        return base * k
    out["solar_land"] = out.apply(sol_row, axis=1)
    out["sunshine_land"] = np.where(out["solar_land"] > 0.05, 0.6, 0.0)
    out = out.sort_values(["date", "hour_n"]).reset_index(drop=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"  → {OUT}  {len(out)}행  {out['date'].min()} ~ {out['date'].max()}")

    # weather_hourly 결측 시각 보완 — ASOS 실측은 덮어쓰지 않음
    if os.path.exists(WSRC):
        w = pd.read_csv(WSRC, encoding="utf-8-sig")
        fill_cols = [c for c in ("temp_land", "humid_land", "wind_spd_land",
                                 "rain_land", "solar_land", "sunshine_land") if c in out.columns]
        add = out[["date", "hour_n"] + fill_cols].copy()
        add = add.rename(columns={c: c + "_fcst" for c in fill_cols})
        m = w.merge(add, on=["date", "hour_n"], how="outer")
        for c in fill_cols:
            if c not in m.columns:
                m[c] = np.nan
            fc = c + "_fcst"
            m[c] = m[c].where(m[c].notna(), m[fc])
        keep = [c for c in w.columns]
        m = m[keep].sort_values(["date", "hour_n"]).reset_index(drop=True)
        tmp = WSRC + ".tmp"
        m.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, WSRC)
        print(f"  → weather_hourly 보완 {w['date'].max()} → {m['date'].max()}  ({len(m)}행)")

    print("=" * 72)


if __name__ == "__main__":
    main()
