# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step3 학습 데이터셋 통합
================================================
목적 : step2 산출물(API·파일데이터)을 단일 시간별 데이터셋으로 병합
기간 : 기본 2021-01-01 ~ 2026-07-30 (--start/--end 로 변경)
출력 : data/processed/smp_dataset_hourly.csv
       logs/step3_dataset_report_{일시}.csv

소스 우선순위 규칙
  · SMP(육지·제주)     : API 단일 소스
  · 수요예측(3종)       : API 단일 소스
  · 전력수요 실적       : 파일데이터(확정치)
  · 태양광·풍력        : 파일데이터(확정치) 우선 → 미보유 구간은 API(잠정치) 보완
  · 공휴일             : API(한국천문연구원 특일정보)
  · 가격결정발전기      : 기존 사내자료(보유 구간에 한함)

버전 : v1.2 (2026-07-31)
실행 : python scripts\\step3_build_dataset.py
       python scripts\\step3_build_dataset.py --start 20230101 --end 20260730
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 미설치 → pip install -r requirements.txt")

INTERIM = os.path.join(BASE_DIR, "data", "interim")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

SRC = {
    "smp": "api_smp_demand_hourly.csv",       # SMP·수요예측
    "pv_file": "file_pv_wind_daily_sum.csv",  # 태양광·풍력(확정)
    "pv_api": "api_pv_hourly.csv",            # 태양광(잠정)
    "demand": "file_demand_hourly.csv",       # 전력수요 실적
    "holiday": "api_holiday.csv",             # 공휴일
    "unit": "hourly_smp_clean.csv",           # 가격결정발전기(사내자료)
    "weather": "weather_hourly.csv",          # 기상(ASOS 권역 가중평균)
}

REPORT = []


def rep(구분, 항목, 값, 비고=""):
    REPORT.append({"구분": 구분, "항목": 항목, "값": str(값), "비고": 비고})
    print(f"  · {항목} : {값} {('- ' + 비고) if 비고 else ''}")


def load(key, required=True):
    p = os.path.join(INTERIM, SRC[key])
    if not os.path.exists(p):
        if required:
            sys.exit(f"[중단] 입력파일 없음 → {p}\n       step2 스크립트 선행 실행 필요")
        print(f"  [건너뜀] {SRC[key]} 없음")
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="step3 학습 데이터셋 통합")
    # 종료일 기본값 : 익일(D+1) — 운영 예측 대상 행을 골격에 포함
    #  · 고정값 사용 시 신규 수집분이 반영되지 않는 문제 방지(v1.1, 2026.8.10)
    #  · 실적 미확정 구간은 타깃 결측 상태로 생성되며, 학습 단계에서 자동 제외됨
    _default_end = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    ap.add_argument("--start", default="20210101", help="시작일 YYYYMMDD")
    ap.add_argument("--end", default=_default_end,
                    help="종료일 YYYYMMDD (기본: 익일 — 예측 대상 포함)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d").strftime("%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y%m%d").strftime("%Y-%m-%d")

    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 70)
    print(" step3 학습 데이터셋 통합 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 대상기간 : {start} ~ {end}")
    print("=" * 70)

    # ── 1. 골격 : 일자 × 24시간 ─────────────────
    print("\n■ ① 시간축 골격 생성")
    days = pd.date_range(start, end, freq="D")
    base = pd.MultiIndex.from_product(
        [days.strftime("%Y-%m-%d"), range(1, 25)], names=["date", "hour_n"]
    ).to_frame(index=False)
    rep("골격", "대상 일수", f"{len(days):,}일")
    rep("골격", "대상 행수", f"{len(base):,}행", "일 24시간")

    # ── 2. SMP·수요예측 ────────────────────────
    print("\n■ ② SMP·수요예측 병합 (API)")
    smp = load("smp")
    smp = smp[(smp["date"] >= start) & (smp["date"] <= end)]
    smp = smp.rename(columns={
        "demand_land": "demand_fcst_land",
        "demand_jeju": "demand_fcst_jeju",
        "demand_total": "demand_fcst_total",
    })
    df = base.merge(smp, on=["date", "hour_n"], how="left")
    rep("SMP", "병합 행수", f"{smp.shape[0]:,}행",
        f"{smp['date'].min()} ~ {smp['date'].max()}")
    rep("SMP", "결측 행수", f"{df['smp_land'].isna().sum():,}행")

    # ── 3. 전력수요 실적 ───────────────────────
    print("\n■ ③ 전력수요 실적 병합 (파일데이터)")
    dem = load("demand", required=False)
    if dem is not None:
        dem = dem.rename(columns={"demand_mwh": "demand_actual"})
        df = df.merge(dem[["date", "hour_n", "demand_actual"]],
                      on=["date", "hour_n"], how="left")
        n_na = int(df["demand_actual"].isna().sum())
        rep("수요실적", "결측 행수", f"{n_na:,}행",
            f"{100 * n_na / len(df):.1f}% — 2026년 구간 미제공")
    else:
        df["demand_actual"] = pd.NA

    # ── 4. 태양광·풍력 (파일 우선 → API 보완) ──
    print("\n■ ④ 태양광·풍력 병합 (파일 우선, API 보완)")
    fv = load("pv_file", required=False)
    if fv is not None:
        fv = fv.rename(columns={
            "pv_mwh_land": "pv_land", "pv_mwh_jeju": "pv_jeju",
            "wind_mwh_land": "wind_land", "wind_mwh_jeju": "wind_jeju",
        })
        cols = ["date", "hour_n", "pv_land", "pv_jeju", "pv_total",
                "wind_land", "wind_jeju", "wind_total"]
        df = df.merge(fv[[c for c in cols if c in fv.columns]],
                      on=["date", "hour_n"], how="left")
        df["pv_source"] = df["pv_total"].notna().map({True: "파일(확정)", False: pd.NA})
        n_file = int(df["pv_total"].notna().sum())
        rep("신재생", "파일데이터 충당", f"{n_file:,}행",
            f"{fv['date'].min()} ~ {fv['date'].max()}")
    else:
        for c in ("pv_land", "pv_jeju", "pv_total", "wind_land", "wind_jeju", "wind_total"):
            df[c] = pd.NA
        df["pv_source"] = pd.NA

    av = load("pv_api", required=False)
    if av is not None:
        av = av.rename(columns={"pv_land": "pv_land_a", "pv_jeju": "pv_jeju_a",
                                "pv_total": "pv_total_a"})
        df = df.merge(av[["date", "hour_n", "pv_land_a", "pv_jeju_a", "pv_total_a"]],
                      on=["date", "hour_n"], how="left")
        m = df["pv_total"].isna() & df["pv_total_a"].notna()
        for tgt, srcc in (("pv_land", "pv_land_a"), ("pv_jeju", "pv_jeju_a"),
                          ("pv_total", "pv_total_a")):
            df.loc[m, tgt] = df.loc[m, srcc]
        df.loc[m, "pv_source"] = "API(잠정)"
        df = df.drop(columns=["pv_land_a", "pv_jeju_a", "pv_total_a"])
        rep("신재생", "API 보완", f"{int(m.sum()):,}행", "2026년 구간")
    n_na = int(df["pv_total"].isna().sum())
    rep("신재생", "태양광 결측", f"{n_na:,}행", f"{100 * n_na / len(df):.1f}%")
    n_na_w = int(df["wind_total"].isna().sum())
    rep("신재생", "풍력 결측", f"{n_na_w:,}행",
        f"{100 * n_na_w / len(df):.1f}% — API 미제공 구간")

    # ── 5. 달력 변수·공휴일 ────────────────────
    print("\n■ ⑤ 달력 변수·공휴일 부여")
    dt = pd.to_datetime(df["date"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dow"] = dt.dt.weekday                    # 0=월 … 6=일
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    hol = load("holiday", required=False)
    if hol is not None:
        h = hol[hol["is_holiday"] == 1][["date", "date_name"]].drop_duplicates("date")
        df = df.merge(h.rename(columns={"date_name": "holiday_name"}),
                      on="date", how="left")
        df["is_holiday"] = df["holiday_name"].notna().astype(int)
    else:
        df["holiday_name"] = pd.NA
        df["is_holiday"] = 0
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["daytype"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).map(
        {True: "휴일", False: "평일"})
    rep("달력", "공휴일 일수", f"{df[df['is_holiday'] == 1]['date'].nunique():,}일")
    rep("달력", "휴일 비중",
        f"{100 * (df['daytype'] == '휴일').mean():.1f}%", "주말 + 공휴일")

    # ── 6. 가격결정발전기(사내자료, 보유 구간) ──
    print("\n■ ⑥ 가격결정발전기 병합 (사내자료)")
    # 소스 우선순위 : ① src_price_setter.csv(추가 수령분 병합본, 2021~)
    #                ② hourly_smp_clean.csv(원본 Excel 기반, 2023~)
    #  · R-1 수령(2026.8.11)으로 2021~2022 구간이 확보되어 우선 참조(v1.2)
    cand = [os.path.join(INTERIM, "src_price_setter.csv"),
            os.path.join(INTERIM, SRC["unit"])]
    up = next((c for c in cand if os.path.exists(c)), None)
    if up:
        u = pd.read_csv(up, encoding="utf-8-sig", low_memory=False)
        if "hour_n" not in u.columns and "hour" in u.columns:
            u["hour_n"] = pd.to_numeric(
                u["hour"].astype(str).str.replace(r"[^0-9]", "", regex=True),
                errors="coerce")
            u = u.dropna(subset=["hour_n"])
            u["hour_n"] = u["hour_n"].astype(int)
        rep("발전기", "참조 소스", os.path.basename(up),
            f"{u['date'].min()} ~ {u['date'].max()}")
        keep = [c for c in ("date", "hour_n", "unit_id", "unit_name", "grp") if c in u.columns]
        u = u[keep].drop_duplicates(subset=["date", "hour_n"])
        df = df.merge(u, on=["date", "hour_n"], how="left")
        n = int(df["unit_id"].notna().sum()) if "unit_id" in df.columns else 0
        rep("발전기", "보유 행수", f"{n:,}행",
            f"{100 * n / len(df):.1f}% — 잔여는 미보유 구간")
    else:
        print("  [건너뜀] 사내자료 없음")

    # ── 6-2. 기상(ASOS 권역 가중평균) ──────────
    print("\n■ ⑥-2 기상 데이터 병합")
    wp = os.path.join(INTERIM, SRC["weather"])
    wcols = []
    if os.path.exists(wp):
        w = pd.read_csv(wp, encoding="utf-8-sig")
        wcols = [c for c in w.columns if c not in ("date", "hour_n")]
        df = df.merge(w, on=["date", "hour_n"], how="left")
        cov = float(df[wcols[0]].notna().mean()) if wcols else 0.0
        rep("기상", "병합 변수", f"{len(wcols)}종", ", ".join(wcols))
        rep("기상", "적용 구간", f"{cov * 100:.1f}%",
            "미확보 구간은 결측 — 학습 시 자동 제외" if cov < 0.99 else "전 구간 확보")
    else:
        print("  [건너뜀] 기상자료 없음 → step2_parse_weather.py 선행 실행 시 반영")

    # ── 7. 정리·저장 ───────────────────────────
    print("\n■ ⑦ 정리·저장")
    order = ["date", "hour_n", "year", "month", "day", "dow", "week_of_year",
             "daytype", "is_weekend", "is_holiday", "holiday_name",
             "smp_land", "smp_jeju",
             "demand_fcst_land", "demand_fcst_jeju", "demand_fcst_total",
             "demand_actual",
             "pv_land", "pv_jeju", "pv_total",
             "wind_land", "wind_jeju", "wind_total", "pv_source",
             "unit_id", "unit_name", "grp"] + wcols
    out = df[[c for c in order if c in df.columns]].sort_values(
        ["date", "hour_n"]).reset_index(drop=True)

    path = os.path.join(PROCESSED, "smp_dataset_hourly.csv")
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)          # 원자적 교체 — 중단 시 기존 파일 보존
    rep("산출", "행수", f"{len(out):,}행")
    rep("산출", "열수", f"{len(out.columns)}개")
    rep("산출", "기간", f"{out['date'].min()} ~ {out['date'].max()}")
    print(f"  → 저장 {path}")

    # ── 8. 변수별 결측 요약 ────────────────────
    print("\n■ ⑧ 변수별 결측 현황")
    print(f"  {'변수':<20}{'결측행':>10}{'결측률':>9}")
    print("  " + "-" * 39)
    for c in out.columns:
        n = int(out[c].isna().sum())
        if n:
            print(f"  {c:<20}{n:>10,}{100 * n / len(out):>8.1f}%")
            REPORT.append({"구분": "결측", "항목": c, "값": f"{n:,}행",
                           "비고": f"{100 * n / len(out):.1f}%"})
    if not out.isna().any().any():
        print("  결측 없음")

    lp = os.path.join(LOG_DIR, f"step3_dataset_report_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(lp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["구분", "항목", "값", "비고"])
        w.writeheader()
        w.writerows(REPORT)
    print(f"\n  → 보고서 {lp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
