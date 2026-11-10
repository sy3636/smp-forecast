# -*- coding: utf-8 -*-
"""
step3_extend_2020.py — 데이터 시작일을 2020-01-01로 앞당김 (팀 작업본, 2026-11-08)
==============================================================================
원칙 : 2021-01-01 이후 기존 행은 한 글자도 바꾸지 않고, 2020년 행만 앞에 붙인다.

2020년 출처
  · SMP·하루전 수요예측 : data/interim/api_smp_demand_hourly.csv (이미 2020부터 보유)
  · 태양광·풍력         : data/interim/file_pv_wind_daily_sum.csv (이미 2017부터 보유)
  · 공휴일              : 한국천문연구원 특일정보 API → data/raw/holiday/2020*.xml
  · 기상(ASOS 8지점)     : 기상청 ASOS 시간자료 API → data/raw/asos/*_2020*.json
  · 국제 연료지수        : FRED (POILDUBUSDM·PNGASJPUSDM·PCOALAUUSDM·EXKOUS)
                          → data/external/fuel_index_fred_2020_20261108.csv
  · 전력수요 실적        : 없음(빈칸) — 모델에서 쓰지 않음

선행 : step1_collect_weather.py --start 20200101 --end 20201231
       step1_collect.py --source holiday --start 20200101 --end 20201231
실행 : python scripts/step3_extend_2020.py --check   (2021년으로 같은 규칙 재현 검사만)
       python scripts/step3_extend_2020.py           (2020년 행 추가)
"""
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
INTERIM = os.path.join(BASE, "data", "interim")
PROCESSED = os.path.join(BASE, "data", "processed")
EXTERNAL = os.path.join(BASE, "data", "external")
FRED_SNAP = os.path.join(EXTERNAL, "fuel_index_fred_2020_20261108.csv")
START, END = "2020-01-01", "2020-12-31"


def rd(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False, **kw)


def save(df, p):
    tmp = p + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, p)


# ------------------------------------------------------------------ 공휴일
def holidays_2020():
    rows = []
    for p in sorted(glob.glob(os.path.join(BASE, "data", "raw", "holiday", "2020*.xml"))):
        for it in ET.parse(p).getroot().iter("item"):
            if it.findtext("isHoliday") == "Y":
                d = it.findtext("locdate")
                rows.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "date_name": it.findtext("dateName"), "is_holiday": 1})
    return pd.DataFrame(rows, columns=["date", "date_name", "is_holiday"])


# ------------------------------------------------------------------ 기상
def weather_2020():
    import step2_parse_weather as pw
    tmp = os.path.join(INTERIM, "_weather_2020_tmp.csv")
    pw.OUT = tmp
    pw.main()
    w = rd(tmp)
    os.remove(tmp)
    return w[(w["date"] >= START) & (w["date"] <= END)]


# ------------------------------------------------------------------ 연료지수
def fuel_2020():
    f = rd(FRED_SNAP)
    f = f[(f["ym"] >= "2020-01") & (f["ym"] <= "2020-12")].copy()
    for c in ("dubai_usd_bbl", "lng_asia_usd_mmbtu", "coal_au_usd_ton", "krw_per_usd"):
        f[c] = f[c].round(4)
    f["idx_oil_krw"] = (f["dubai_usd_bbl"] * f["krw_per_usd"]).round(2)
    f["idx_lng_krw"] = (f["lng_asia_usd_mmbtu"] * f["krw_per_usd"]).round(2)
    f["idx_coal_krw"] = (f["coal_au_usd_ton"] * f["krw_per_usd"]).round(2)
    return f


# ------------------------------------------------------------------ 데이터셋 행
def build_rows(start, end, hol, wx):
    days = pd.date_range(start, end, freq="D")
    g = pd.MultiIndex.from_product([days.strftime("%Y-%m-%d"), range(1, 25)], names=["date", "hour_n"]).to_frame(index=False)
    dt = pd.to_datetime(g["date"])
    g["year"], g["month"], g["day"] = dt.dt.year, dt.dt.month, dt.dt.day
    g["dow"] = dt.dt.dayofweek
    g["week_of_year"] = dt.dt.isocalendar().week.astype(int).to_numpy()
    h = hol[hol["is_holiday"] == 1].drop_duplicates("date").set_index("date")["date_name"]
    g["is_weekend"] = (g["dow"] >= 5).astype(int)
    g["is_holiday"] = g["date"].isin(h.index).astype(int)
    g["holiday_name"] = g["date"].map(h)
    g["daytype"] = np.where((g["is_weekend"] == 1) | (g["is_holiday"] == 1), "휴일", "평일")

    s = rd(os.path.join(INTERIM, "api_smp_demand_hourly.csv")).rename(columns={
        "demand_land": "demand_fcst_land", "demand_jeju": "demand_fcst_jeju", "demand_total": "demand_fcst_total"})
    g = g.merge(s, on=["date", "hour_n"], how="left")
    dm = rd(os.path.join(INTERIM, "file_demand_hourly.csv")).rename(columns={"demand_mwh": "demand_actual"})
    g = g.merge(dm, on=["date", "hour_n"], how="left")
    pv = rd(os.path.join(INTERIM, "file_pv_wind_daily_sum.csv")).rename(columns={
        "pv_mwh_land": "pv_land", "pv_mwh_jeju": "pv_jeju", "wind_mwh_land": "wind_land", "wind_mwh_jeju": "wind_jeju"})
    g = g.merge(pv, on=["date", "hour_n"], how="left")
    g["pv_source"] = np.where(g["pv_land"].notna(), "파일(확정)", None)
    g = g.merge(wx, on=["date", "hour_n"], how="left")
    return g


def compare(new, old, cols):
    m = new.merge(old, on=["date", "hour_n"], suffixes=("_n", "_o"))
    bad = {}
    for c in cols:
        a, b = m[c + "_n"], m[c + "_o"]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            diff = ~(np.isclose(a, b, rtol=0, atol=1e-6) | (a.isna() & b.isna()))
        else:
            diff = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
        if diff.any():
            bad[c] = int(diff.sum())
    return len(m), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="2021년을 같은 규칙으로 만들어 기존 데이터셋과 비교만")
    a = ap.parse_args()

    ds_path = os.path.join(PROCESSED, "smp_dataset_hourly.csv")
    ds = rd(ds_path)
    cols = list(ds.columns)

    if a.check:
        hol = rd(os.path.join(INTERIM, "api_holiday.csv"))
        wx = rd(os.path.join(INTERIM, "weather_hourly.csv"))
        new = build_rows("2021-01-01", "2021-12-31", hol, wx)[cols]
        n, bad = compare(new, ds, [c for c in cols if c not in ("date", "hour_n")])
        print(f"[검사] 2021년 {n}행 비교 — 다른 열: {bad if bad else '없음'}")
        return

    if ds["date"].min() <= END:
        sys.exit(f"[중단] 데이터셋이 이미 {ds['date'].min()}부터 있음 — 중복 추가 방지")

    # 1) 공휴일
    hp = os.path.join(INTERIM, "api_holiday.csv")
    hol_old = rd(hp)
    hol20 = holidays_2020()
    hol = pd.concat([hol20, hol_old[hol_old["date"] > END]], ignore_index=True)
    save(hol, hp)
    print(f"· 공휴일 2020 : {len(hol20)}일 추가")

    # 2) 기상
    wp = os.path.join(INTERIM, "weather_hourly.csv")
    wx_old = rd(wp)
    wx20 = weather_2020()[list(wx_old.columns)]
    save(pd.concat([wx20, wx_old[wx_old["date"] > END]], ignore_index=True), wp)
    print(f"· 기상 2020 : {len(wx20)}행 추가 (temp_land 결측 {wx20['temp_land'].isna().sum()}행)")

    # 3) 연료지수
    fp = os.path.join(INTERIM, "fuel_index_monthly.csv")
    f_old = rd(fp)
    f20 = fuel_2020()[list(f_old.columns)]
    save(pd.concat([f20, f_old[f_old["ym"] > "2020-12"]], ignore_index=True), fp)
    print(f"· 연료지수 2020 : {len(f20)}개월 추가")

    # 4) 데이터셋
    rows = build_rows(START, END, hol, wx20)[cols]
    save(pd.concat([rows, ds], ignore_index=True), ds_path)
    print(f"· 데이터셋 2020 : {len(rows)}행 추가 → 총 {len(rows) + len(ds)}행")
    print(f"  SMP 결측 {rows['smp_land'].isna().sum()} · 수요예측 결측 {rows['demand_fcst_land'].isna().sum()} · "
          f"태양광 결측 {rows['pv_land'].isna().sum()} · 기온 결측 {rows['temp_land'].isna().sum()}")


if __name__ == "__main__":
    main()
