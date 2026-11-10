# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step2 파일데이터 파싱·통합
================================================
목적 : 공공데이터포털 파일데이터(CSV)의 이질적 3개 형식을 단일 스키마로 통합
입력 : data/raw/file/*.csv (전 파일 CP949)
출력 : data/interim/file_pv_wind_hourly.csv  (지역별 시간별 태양광·풍력)
       data/interim/file_pv_wind_daily_sum.csv (육지·제주 집계)
       data/interim/file_demand_hourly.csv   (시간별 전국 전력수요)
       logs/step2_parse_file_{일시}.csv      (검증 결과)
버전 : v1.0 (2026-07-31)

대응 형식:
  A형(가로) : 거래일자, 거래시간, 지역, 태양광 발전량(MWh), 풍력 발전량(MWh)   [2017~2023.11]
  B형(세로) : 거래일자/거래일, 거래시간, 지역, 연료원, 전력거래량(MWh)          [2023.12~2025]
  D형(수요) : 날짜, 1시, 2시, … 24시                                            [2023~2025]

제외 : 자사 실적자료(한국중부발전 신재생에너지 발전량) — 스키마 상이

실행 : python scripts\\step2_parse_file.py
"""

import csv
import glob
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 미설치 → pip install -r requirements.txt")

RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "file")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")

ENCODINGS = ("cp949", "utf-8-sig", "utf-8", "euc-kr")
EXCLUDE_KEYWORDS = ("중부발전",)      # 자사 자료 등 스키마 상이분

CHECKS = []


def log_check(구분, 항목, 결과, 상세=""):
    CHECKS.append({"구분": 구분, "검증항목": 항목, "결과": 결과, "상세": 상세})
    mark = {"정상": "○", "주의": "△", "오류": "×"}.get(결과, "-")
    print(f"  [{mark}] {항목} : {결과} {('- ' + 상세) if 상세 else ''}")


def read_csv_any(path):
    """인코딩 자동 판별 읽기."""
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False), enc
        except Exception:
            continue
    return None, None


def norm_cols(df):
    """컬럼명 정규화 — 공백 제거, 표기 변형 통일."""
    ren = {}
    for c in df.columns:
        k = re.sub(r"\s+", "", str(c))
        k = k.replace("Mwh", "MWh")
        if k in ("거래일자", "거래일", "날짜", "일자"):
            k = "date"
        elif k in ("거래시간", "시간"):
            k = "hour_n"
        elif k in ("지역", "지역명", "광역시도"):
            k = "region"
        elif k in ("연료원",):
            k = "fuel"
        elif k in ("전력거래량(MWh)", "발전량(MWh)", "거래량(MWh)"):
            k = "amount"
        elif k.startswith("태양광"):
            k = "pv_mwh"
        elif k.startswith("풍력"):
            k = "wind_mwh"
        ren[c] = k
    return df.rename(columns=ren)


def detect_format(cols):
    s = set(cols)
    if {"date", "hour_n", "region", "fuel", "amount"} <= s:
        return "B"
    if {"date", "hour_n", "region"} <= s and ("pv_mwh" in s or "wind_mwh" in s):
        return "A"
    if "date" in s and any(re.fullmatch(r"\d{1,2}시", str(c)) for c in cols):
        return "D"
    return "?"


def to_date(s):
    return pd.to_datetime(s.astype(str).str.strip(), errors="coerce")


# ------------------------------------------------------------------
# 1. 태양광·풍력 (A형 + B형)
# ------------------------------------------------------------------
def build_pv_wind(files):
    print("\n■ 태양광·풍력 파일데이터 파싱")
    frames = []
    for path, df, enc, fmt in files:
        name = os.path.basename(path)
        if fmt == "A":
            out = df[["date", "hour_n", "region"]].copy()
            out["pv_mwh"] = pd.to_numeric(df.get("pv_mwh"), errors="coerce")
            out["wind_mwh"] = pd.to_numeric(df.get("wind_mwh"), errors="coerce")
        elif fmt == "B":
            df = df.copy()
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["fuel"] = df["fuel"].astype(str).str.strip()
            piv = df.pivot_table(index=["date", "hour_n", "region"],
                                 columns="fuel", values="amount",
                                 aggfunc="sum").reset_index()
            piv = piv.rename(columns={"태양광": "pv_mwh", "풍력": "wind_mwh"})
            for c in ("pv_mwh", "wind_mwh"):
                if c not in piv.columns:
                    piv[c] = pd.NA
            out = piv[["date", "hour_n", "region", "pv_mwh", "wind_mwh"]]
        else:
            continue
        out = out.copy()
        out["date"] = to_date(out["date"]).dt.strftime("%Y-%m-%d")
        out["hour_n"] = pd.to_numeric(out["hour_n"], errors="coerce").astype("Int64")
        out["region"] = out["region"].astype(str).str.strip()
        out["_src"] = name
        frames.append(out)
        print(f"  · {fmt}형 {len(out):>8,}행  {name[:45]}")

    if not frames:
        log_check("태양광·풍력", "원문 파싱", "오류", "대상 파일 없음")
        return None

    df = pd.concat(frames, ignore_index=True).dropna(subset=["date", "hour_n"])

    n0 = len(df)
    df = df.drop_duplicates(subset=["date", "hour_n", "region"], keep="first")
    log_check("태양광·풍력", "중복 제거", "정상", f"{n0:,}행 → {len(df):,}행")

    # 연속성
    d = sorted(df["date"].unique())
    full = pd.date_range(d[0], d[-1]).strftime("%Y-%m-%d")
    miss = sorted(set(full) - set(d))
    log_check("태양광·풍력", "일자 연속성", "정상" if not miss else "주의",
              f"{d[0]} ~ {d[-1]} / {len(d):,}일" + (f" / 결손 {len(miss)}일" if miss else " / 결손 0일"))

    # 시간 완비
    c = df.groupby("date")["hour_n"].nunique()
    bad = c[c != 24]
    log_check("태양광·풍력", "24시간 완비", "정상" if bad.empty else "주의",
              "전 일자 완비" if bad.empty else f"불완전 {len(bad)}일")

    # 지역 수
    r = df.groupby("date")["region"].nunique()
    mode_n = int(r.mode().iloc[0])
    bad_r = r[r != mode_n]
    log_check("태양광·풍력", "일자별 지역 수", "정상" if bad_r.empty else "주의",
              f"표준 {mode_n}개" + ("" if bad_r.empty else f" / 상이 {len(bad_r)}일(초기 구간 축소 운영 추정)"))

    out = df[["date", "hour_n", "region", "pv_mwh", "wind_mwh"]].sort_values(
        ["date", "hour_n", "region"]).reset_index(drop=True)
    p1 = os.path.join(INTERIM, "file_pv_wind_hourly.csv")
    out.to_csv(p1, index=False, encoding="utf-8-sig")
    print(f"  → 저장 {p1} ({len(out):,}행)")

    # 육지·제주 집계
    out2 = out.copy()
    out2["구분"] = out2["region"].apply(lambda x: "jeju" if "제주" in str(x) else "land")
    agg = out2.pivot_table(index=["date", "hour_n"], columns="구분",
                           values=["pv_mwh", "wind_mwh"], aggfunc="sum").reset_index()
    agg.columns = ["date", "hour_n"] + [f"{a}_{b}" for a, b in agg.columns[2:]]
    for c in ("pv_mwh_land", "pv_mwh_jeju", "wind_mwh_land", "wind_mwh_jeju"):
        if c not in agg.columns:
            agg[c] = 0.0
    agg["pv_total"] = agg["pv_mwh_land"].fillna(0) + agg["pv_mwh_jeju"].fillna(0)
    agg["wind_total"] = agg["wind_mwh_land"].fillna(0) + agg["wind_mwh_jeju"].fillna(0)
    p2 = os.path.join(INTERIM, "file_pv_wind_daily_sum.csv")
    agg.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"  → 저장 {p2} ({len(agg):,}행)")
    return out


# ------------------------------------------------------------------
# 2. 전력수요 (D형)
# ------------------------------------------------------------------
def build_demand(files):
    print("\n■ 전력수요 파일데이터 파싱")
    frames = []
    for path, df, enc, fmt in files:
        if fmt != "D":
            continue
        name = os.path.basename(path)
        hcols = [c for c in df.columns if re.fullmatch(r"\d{1,2}시", str(c))]
        m = df[["date"] + hcols].copy()
        m = m.melt(id_vars="date", var_name="hour", value_name="demand_mwh")
        m["hour_n"] = m["hour"].str.replace("시", "", regex=False).astype(int)
        m["date"] = to_date(m["date"]).dt.strftime("%Y-%m-%d")
        m["demand_mwh"] = pd.to_numeric(m["demand_mwh"], errors="coerce")
        m = m.dropna(subset=["date"])
        frames.append(m[["date", "hour_n", "demand_mwh"]])
        print(f"  · D형 {len(m):>8,}행  {name[:45]}")

    if not frames:
        log_check("전력수요", "원문 파싱", "오류", "대상 파일 없음")
        return None

    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(subset=["date", "hour_n"], keep="first")
    log_check("전력수요", "중복 제거", "정상", f"{n0:,}행 → {len(df):,}행")

    d = sorted(df["date"].unique())
    full = pd.date_range(d[0], d[-1]).strftime("%Y-%m-%d")
    miss = sorted(set(full) - set(d))
    log_check("전력수요", "일자 연속성", "정상" if not miss else "주의",
              f"{d[0]} ~ {d[-1]} / {len(d):,}일 / 결손 {len(miss)}일")

    c = df.groupby("date")["hour_n"].nunique()
    bad = c[c != 24]
    log_check("전력수요", "24시간 완비", "정상" if bad.empty else "주의",
              "전 일자 완비" if bad.empty else f"불완전 {len(bad)}일")

    na = int(df["demand_mwh"].isna().sum())
    log_check("전력수요", "결측·범위", "정상" if na == 0 else "주의",
              f"결측 {na}행 / 범위 {df['demand_mwh'].min():,.0f}~{df['demand_mwh'].max():,.0f} MWh")

    out = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    p = os.path.join(INTERIM, "file_demand_hourly.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  → 저장 {p} ({len(out):,}행)")
    return out


# ------------------------------------------------------------------
def main():
    os.makedirs(INTERIM, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 70)
    print(" step2 파일데이터 파싱·통합 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    paths = sorted(glob.glob(os.path.join(RAW_FILE, "*.csv")))
    if not paths:
        sys.exit(f"[중단] 원문 없음 → {RAW_FILE}")

    files, skipped = [], []
    for p in paths:
        name = os.path.basename(p)
        if any(k in name for k in EXCLUDE_KEYWORDS):
            skipped.append((name, "제외 대상(자사 자료)"))
            continue
        df, enc = read_csv_any(p)
        if df is None:
            skipped.append((name, "인코딩 판별 실패"))
            continue
        df = norm_cols(df)
        fmt = detect_format(df.columns)
        if fmt == "?":
            skipped.append((name, f"형식 미인식 {list(df.columns)[:4]}"))
            continue
        files.append((p, df, enc, fmt))

    print(f"\n■ 대상 파일 {len(files)}건 / 제외 {len(skipped)}건")
    for n, r in skipped:
        print(f"  · 제외 : {n[:45]} — {r}")
    log_check("공통", "파일 인식", "정상" if files else "오류",
              f"대상 {len(files)}건 / 제외 {len(skipped)}건")

    build_pv_wind(files)
    build_demand(files)

    path = os.path.join(LOG_DIR, f"step2_parse_file_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["구분", "검증항목", "결과", "상세"])
        w.writeheader()
        w.writerows(CHECKS)

    n_ng = sum(1 for c in CHECKS if c["결과"] == "오류")
    n_wn = sum(1 for c in CHECKS if c["결과"] == "주의")
    print("\n" + "=" * 70)
    print(f" 검증 종합 : 총 {len(CHECKS)}항목 / 오류 {n_ng} / 주의 {n_wn}")
    print(f" 검증기록 : {path}")
    print("=" * 70)
    sys.exit(1 if n_ng else 0)


if __name__ == "__main__":
    main()
