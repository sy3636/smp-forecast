# -*- coding: utf-8 -*-
"""
collect_recent.py — 학습 데이터(smp_dataset_hourly.csv) 이후 기간을 공공데이터 API로 보충
==============================================================================
  · SMP·수요예측 : 한국전력거래소_계통한계가격 및 수요예측 — 데이터셋 마지막 SMP 다음 날 ~ 공표분 끝까지
  · 기온         : 기상청 ASOS 시간자료 7개 지점 가중평균 (scripts/step1_collect_weather.py 와 같은 지점·가중,
                   ASOS 0~23시 → 전력시장 1~24시 +1 보정)
  · 인증키       : config/.env 의 KPX_API_KEY · KMA_API_KEY (같은 키)
  · 개발계정 트래픽 : SMP API 하루 100건 — 하루 1건씩 호출하므로 한 달 보충에 약 31건

실행
  python experiments/ensemble/collect_recent.py                     # 공표된 마지막 날까지
  python experiments/ensemble/collect_recent.py --end 2026-10-06    # 이 날짜까지만 (팀원 재현용)
산출 (data/interim/ — 저장소에 올리지 않음)
  api_smp_recent.csv  — date, hour_n, smp_land, smp_jeju, demand_fcst_land, demand_fcst_jeju, demand_fcst_total
  asos_temp_recent.csv — date, hour_n, temp_land
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from common import BASE, DATA, rd

SMP_URL = "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
ASOS_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
LAND_STATIONS = {"108": 0.30, "112": 0.10, "133": 0.13, "143": 0.13, "156": 0.10, "159": 0.17, "105": 0.07}


def env_key(name):
    for line in open(BASE / "config" / ".env", encoding="utf-8"):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"config/.env 에 {name} 가 없습니다")


def call(url, params):
    q = urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise SystemExit("하루 호출 한도 초과(429) — 내일 다시 실행하세요")
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"호출 실패: {url}")


def items(j):
    body = j.get("response", {}).get("body", {})
    it = (body.get("items") or {}).get("item") or []
    return it if isinstance(it, list) else [it]


def collect_smp(key, start, end):
    rows = []
    d = start
    while d <= end:
        it = items(call(SMP_URL, {"serviceKey": key, "pageNo": 1, "numOfRows": 48, "dataType": "JSON",
                                  "date": f"{d:%Y%m%d}"}))
        for x in it:
            rows.append({"date": f"{d:%Y-%m-%d}", "hour_n": int(x["hour"]), "area": x["areaName"],
                         "smp": x["smp"], "mlfd": x["mlfd"], "jlfd": x["jlfd"], "slfd": x["slfd"]})
        print(f"  SMP {d}: {len(it)}행", flush=True)
        d += timedelta(days=1)
    df = pd.DataFrame(rows)
    land = df[df["area"] == "육지"].rename(columns={"smp": "smp_land"})
    jeju = df[df["area"] == "제주"][["date", "hour_n", "smp"]].rename(columns={"smp": "smp_jeju"})
    out = land.merge(jeju, on=["date", "hour_n"], how="left")
    out = out.rename(columns={"mlfd": "demand_fcst_land", "jlfd": "demand_fcst_jeju", "slfd": "demand_fcst_total"})
    return out[["date", "hour_n", "smp_land", "smp_jeju", "demand_fcst_land", "demand_fcst_jeju", "demand_fcst_total"]]


def collect_temp(key, start, end):
    frames = []
    for stn, w in LAND_STATIONS.items():
        it, page = [], 1
        while True:                                            # 한 번에 최대 999행 — 기간이 길면 여러 쪽으로 나눠 받음
            got = items(call(ASOS_URL, {"serviceKey": key, "pageNo": page, "numOfRows": 999, "dataType": "JSON",
                                        "dataCd": "ASOS", "dateCd": "HR", "startDt": f"{start:%Y%m%d}", "startHh": "00",
                                        "endDt": f"{end:%Y%m%d}", "endHh": "23", "stnIds": stn}))
            it += got
            if len(got) < 999:
                break
            page += 1
        t = pd.DataFrame([{"tm": x["tm"], "ta": pd.to_numeric(x.get("ta"), errors="coerce")} for x in it])
        t["stn"], t["w"] = stn, w
        frames.append(t)
        print(f"  ASOS {stn}: {len(t)}행", flush=True)
    a = pd.concat(frames)
    tm = pd.to_datetime(a["tm"])
    a["date"] = tm.dt.strftime("%Y-%m-%d")
    a["hour_n"] = tm.dt.hour + 1                                  # 0~23시 → 1~24시
    a = a.dropna(subset=["ta"])
    g = a.groupby(["date", "hour_n"]).apply(lambda x: np.average(x["ta"], weights=x["w"]), include_groups=False)
    return g.rename("temp_land").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="이 날짜(YYYY-MM-DD)까지만 받기 — 기본은 공표된 마지막 날까지")
    args = ap.parse_args()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    yesterday = date.today() - timedelta(days=1)

    key = env_key("KPX_API_KEY")
    d = rd(DATA / "processed" / "smp_dataset_hourly.csv", usecols=["date", "smp_land"])
    last_smp = d.loc[d["smp_land"].notna(), "date"].max()
    start = datetime.strptime(last_smp, "%Y-%m-%d").date() + timedelta(days=1)
    print(f"데이터셋 마지막 SMP {last_smp} → {start} 부터 보충")
    smp = collect_smp(key, start, end or date.today() + timedelta(days=1))
    smp.to_csv(DATA / "interim" / "api_smp_recent.csv", index=False, encoding="utf-8-sig")
    print(f"저장 api_smp_recent.csv : {smp['date'].min()} ~ {smp['date'].max()} ({len(smp)}행)")

    # weather_hourly.csv 의 마지막 며칠은 단기예보로 채운 값(실측과 평균 1.2℃ 차이) → 그 구간부터 ASOS 실측으로 덮어씀
    #  (2025-03·2026-08 대조 시 이 방식의 실측과 기존 파일 차이 0.000℃)
    t0 = start - timedelta(days=2)
    temp = collect_temp(env_key("KMA_API_KEY"), t0, min(end, yesterday) if end else yesterday)
    temp.to_csv(DATA / "interim" / "asos_temp_recent.csv", index=False, encoding="utf-8-sig")
    print(f"저장 asos_temp_recent.csv : {temp['date'].min()} ~ {temp['date'].max()} ({len(temp)}행)")


if __name__ == "__main__":
    main()
