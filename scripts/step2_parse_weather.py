# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step2 기상 데이터 전처리·권역 가중평균
================================================
목적 : 지점별 ASOS 원문을 시간별 전국·제주 대표 기상값으로 집약
입력 : data/raw/asos/{지점}_{YYYYMM}.json
       (대안) data/external/asos_*.csv — 기상자료개방포털 수동 다운로드분
출력 : data/interim/weather_hourly.csv
       logs/step2_weather_report_{일시}.csv

처리 원칙
  · 시각 정합 : ASOS 0~23시 → 전력시장 거래시간 1~24시로 +1 보정
  · 권역 집약 : 육지는 전력수요 비중 가중평균, 제주는 184번 단독
  · 결측 처리 : 지점 결측 시 잔여 지점 가중치 재정규화(전 지점 결측만 NaN)
  · 이상치    : 기온 -40~45℃, 습도 0~100%, 풍속 0~60m/s 범위 외 제거

산출 변수
  temp_land, temp_jeju      기온(℃)
  humid_land, humid_jeju    상대습도(%)
  wind_spd_land             풍속(m/s)
  solar_land, solar_jeju    일사량(MJ/m²)
  sunshine_land             일조시간(hr)
  rain_land                 강수량(mm)

버전 : v1.0 (2026-08-05)
실행 : python scripts\\step2_parse_weather.py
"""

import csv
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step1_collect_weather import STATIONS, LAND_IDS  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "asos")
EXTERNAL = os.path.join(BASE_DIR, "data", "external")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUT = os.path.join(INTERIM, "weather_hourly.csv")

# ASOS 원항목 → 표준 변수명
FIELDS = {
    "ta": "temp",        # 기온(℃)
    "hm": "humid",       # 상대습도(%)
    "ws": "wind_spd",    # 풍속(m/s)
    "icsr": "solar",     # 일사량(MJ/m²)
    "ss": "sunshine",    # 일조시간(hr)
    "rn": "rain",        # 강수량(mm)
}
# 유효범위(이상치 판정)
RANGE = {"temp": (-40, 45), "humid": (0, 100), "wind_spd": (0, 60),
         "solar": (0, 5), "sunshine": (0, 1), "rain": (0, 200)}

REPORT = []


def rep(구분, 항목, 값, 비고=""):
    REPORT.append({"구분": 구분, "항목": 항목, "값": str(값), "비고": 비고})
    print(f"  · {항목} : {값}" + (f"  ({비고})" if 비고 else ""))


# ------------------------------------------------------------------
def load_raw():
    """API 원문(JSON) 적재."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        return pd.DataFrame(), 0
    recs = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        body = (j.get("response") or {}).get("body") or {}
        items = (body.get("items") or {}).get("item") or []
        if isinstance(items, dict):
            items = [items]
        recs.extend(items)
    return pd.DataFrame(recs), len(files)


def load_external():
    """기상자료개방포털 수동 다운로드 CSV 적재(대안 경로).
    · 컬럼 예: 지점, 지점명, 일시, 기온(°C), 강수량(mm), 풍속(m/s), 습도(%), 일조(hr), 일사(MJ/m2)"""
    files = sorted(glob.glob(os.path.join(EXTERNAL, "asos_*.csv")))
    if not files:
        return pd.DataFrame(), 0
    ko = {"지점": "stnId", "일시": "tm", "기온": "ta", "습도": "hm",
          "풍속": "ws", "일사": "icsr", "일조": "ss", "강수량": "rn"}
    dfs = []
    for p in files:
        for enc in ("cp949", "utf-8-sig", "utf-8"):
            try:
                d = pd.read_csv(p, encoding=enc)
                break
            except Exception:
                d = None
        if d is None:
            continue
        ren = {}
        for c in d.columns:
            for k, v in ko.items():
                if c.startswith(k):
                    ren[c] = v
                    break
        dfs.append(d.rename(columns=ren))
    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), len(files)


# ------------------------------------------------------------------
def main():
    os.makedirs(INTERIM, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 78)
    print(" step2 기상 데이터 전처리·권역 가중평균 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    # ── ① 적재 ─────────────────────────────────
    print("\n■ ① 원자료 적재")
    api_df, n_api = load_raw()
    ext_df, n_ext = load_external()
    rep("적재", "API 원문", f"{n_api}개 파일 / {len(api_df):,}행")
    rep("적재", "포털 CSV", f"{n_ext}개 파일 / {len(ext_df):,}행")

    df = pd.concat([d for d in (api_df, ext_df) if not d.empty], ignore_index=True)
    if df.empty:
        sys.exit("[중단] 기상 원자료 없음 → step1_collect_weather.py 선행 실행 필요")

    # ── ② 정규화 ───────────────────────────────
    print("\n■ ② 정규화·시각 정합")
    df["stnId"] = df["stnId"].astype(str).str.strip()
    df["_ts"] = pd.to_datetime(df["tm"], errors="coerce")
    df = df.dropna(subset=["_ts"])

    for src, dst in FIELDS.items():
        df[dst] = pd.to_numeric(df.get(src), errors="coerce") if src in df.columns else np.nan

    # ASOS 규약상 빈값이 '0'을 의미하는 항목 보정
    #  · rn(강수량)   : 무강수 시 빈값
    #  · icsr(일사량) : 야간 빈값
    #  · ss(일조시간) : 일조 없음 시 빈값
    for v in ("rain", "solar", "sunshine"):
        if v in df.columns:
            n = int(df[v].isna().sum())
            df[v] = df[v].fillna(0.0)
            rep("정규화", f"{v} 빈값→0", f"{n:,}건", "ASOS 규약(무강수·야간)")

    n0 = len(df)
    df = df.drop_duplicates(subset=["stnId", "_ts"], keep="last")
    rep("정규화", "중복 제거", f"{n0 - len(df):,}행", f"잔여 {len(df):,}행")

    # 시각 정합: ASOS 0~23시 → 전력시장 거래시간 1~24시
    df["date"] = df["_ts"].dt.strftime("%Y-%m-%d")
    df["hour_n"] = df["_ts"].dt.hour + 1
    rep("정규화", "시각 보정", "ASOS 0~23시 → 거래시간 1~24시", "+1 오프셋 적용")
    rep("정규화", "수집 기간", f"{df['date'].min()} ~ {df['date'].max()}")
    rep("정규화", "수집 지점", f"{df['stnId'].nunique()}개",
        ", ".join(sorted(df["stnId"].unique())[:10]))

    # ── ③ 이상치 ───────────────────────────────
    print("\n■ ③ 이상치 처리")
    for v, (lo, hi) in RANGE.items():
        if v not in df.columns:
            continue
        bad = ((df[v] < lo) | (df[v] > hi)).sum()
        if bad:
            df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan
        rep("이상치", f"{v} 범위 외", f"{bad:,}건", f"허용 {lo}~{hi}")

    # ── ④ 권역 집약 ────────────────────────────
    print("\n■ ④ 권역 가중평균")
    df["_w"] = df["stnId"].map({k: v["weight"] for k, v in STATIONS.items()}).fillna(0)
    land = df[df["stnId"].isin(LAND_IDS)]
    jeju = df[df["stnId"] == "184"]

    def wavg(g, col):
        """결측 지점을 제외하고 가중치 재정규화 후 평균."""
        m = g[col].notna()
        w = g.loc[m, "_w"]
        return np.nan if w.sum() == 0 else float((g.loc[m, col] * w).sum() / w.sum())

    out = None
    for col in FIELDS.values():
        if col not in df.columns:
            continue
        a = (land.groupby(["date", "hour_n"], sort=False)
             .apply(lambda g: wavg(g, col), include_groups=False)
             .rename(f"{col}_land").reset_index())
        b = (jeju.groupby(["date", "hour_n"], sort=False)[col].mean()
             .rename(f"{col}_jeju").reset_index())
        m = a.merge(b, on=["date", "hour_n"], how="outer")
        out = m if out is None else out.merge(m, on=["date", "hour_n"], how="outer")

    out = out.sort_values(["date", "hour_n"]).reset_index(drop=True)

    # 육지 대표성 낮은 변수는 제주 열 제거(관측 1지점으로 대표성 확보 곤란한 항목 정리)
    drop = [c for c in ("wind_spd_jeju", "sunshine_jeju", "rain_jeju") if c in out.columns]
    out = out.drop(columns=drop)

    rep("집약", "산출 행수", f"{len(out):,}행")
    rep("집약", "산출 변수", f"{len(out.columns) - 2}종",
        ", ".join([c for c in out.columns if c not in ("date", "hour_n")]))
    for c in out.columns:
        if c in ("date", "hour_n"):
            continue
        rep("결측", c, f"{out[c].isna().mean() * 100:.1f}%")

    # ── ⑤ 저장 ─────────────────────────────────
    print("\n■ ⑤ 저장")
    tmp = OUT + ".tmp"
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, OUT)          # 원자적 교체 — 중단 시 기존 파일 보존
    print(f"  → 저장 {OUT}")

    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    lp = os.path.join(LOG_DIR, f"step2_weather_report_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["구분", "항목", "값", "비고"])
        w.writeheader()
        w.writerows(REPORT)
    print(f"  → 검증보고 {lp}")

    print("=" * 78)
    print(f" 기상 데이터 {len(out):,}행 확보 "
          f"({out['date'].min()} ~ {out['date'].max()})")
    print(" ※ 다음 단계 : python scripts\\step3_build_dataset.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
