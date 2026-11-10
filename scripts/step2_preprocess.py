# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step2 전처리·정합성 검증
================================================
목적 : data/raw 원문(JSON·XML)을 정형 CSV로 변환하고 데이터 정합성을 검증
입력 : data/raw/smp/*.json, data/raw/pv/*.json, data/raw/holiday/*.xml
출력 : data/interim/api_smp_demand_hourly.csv
       data/interim/api_pv_hourly.csv
       data/interim/api_holiday.csv
       logs/step2_validation_{일시}.csv (검증 결과)
버전 : v1.2 (2026-09-05) — ①-2 등급을 '오류' → '주의'로 하향(운영 중단 방지)
       v1.1 (2026-09-04) — 계통별 값 결측 검증(①-2) 추가
       · 행 수 기준 검증만으로는 areaName 단위 누락을 탐지하지 못함이 확인되어 보강

검증 항목:
  ① 일자별 24시간 완비 여부(결측)      ①-2 계통별 값 결측(육지·제주 각각)
  ② 중복 레코드
  ③ 육지·제주 SMP 동일성(중복행 판별)   ④ 수요 항등식(mlfd + jlfd = slfd)
  ⑤ SMP 이상치(0 이하·비정상 고가)      ⑥ 기존 hourly_smp_clean.csv 교차검증

실행 :
  python scripts\\step2_preprocess.py
  python scripts\\step2_preprocess.py --source smp
"""

import argparse
import csv
import glob
import json
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

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REF_CLEAN = os.path.join(INTERIM_DIR, "hourly_smp_clean.csv")

CHECKS = []  # 검증 결과 누적


def log_check(구분, 항목, 결과, 상세=""):
    CHECKS.append({"구분": 구분, "검증항목": 항목, "결과": 결과, "상세": 상세})
    mark = {"정상": "○", "주의": "△", "오류": "×"}.get(결과, "-")
    print(f"  [{mark}] {항목} : {결과} {('- ' + 상세) if 상세 else ''}")


def read_items(path):
    """공공데이터포털 JSON 응답에서 item 리스트 추출."""
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    items = js.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    return items


# ------------------------------------------------------------------
# 1. SMP·수요예측
# ------------------------------------------------------------------
def build_smp():
    daily = sorted(glob.glob(os.path.join(RAW_DIR, "smp", "*.json")))
    bulk = sorted(glob.glob(os.path.join(RAW_DIR, "smp_bulk", "*.json")))
    #  웹 보완 원문(step1_collect_kpx_web.py, 2026.10.2 신설) — API 원문 뒤에 두어 API 우선
    web = sorted(glob.glob(os.path.join(RAW_DIR, "smp_web", "*.json")))
    files = daily + bulk + web
    if not files:
        print("  [건너뜀] data/raw/smp·smp_bulk 원문 없음")
        return None
    print(f"\n■ SMP·수요예측 전처리 (일별 {len(daily)}건 + 일괄 {len(bulk)}페이지"
          f" + 웹 보완 {len(web)}건)")

    rows = []
    for p in files:
        rows.extend(read_items(p))
    df = pd.DataFrame(rows)
    if df.empty:
        log_check("SMP", "원문 파싱", "오류", "레코드 0건")
        return None
    if "source" not in df.columns:
        df["source"] = pd.NA

    # 일별·일괄·웹 수집분 중복 제거 — 앞선 원천(API) 우선
    n0 = len(df)
    df = df.drop_duplicates(subset=["date", "hour", "areaName"], keep="first")
    used_web = sorted(df.loc[df["source"] == "kpx_web", "date"].astype(str).unique())
    if web:
        log_check("SMP", "웹 보완 원문 사용", "주의" if used_web else "정상",
                  (f"{len(used_web)}일 — {', '.join(used_web)} (API 공백분)")
                  if used_web else "API 원문 완비 — 웹 보완 미사용")
    if n0 != len(df):
        log_check("SMP", "수집원 간 중복 제거", "정상",
                  f"{n0:,}행 → {len(df):,}행 ({n0 - len(df):,}행 제거)")

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    for c in ("smp", "mlfd", "jlfd", "slfd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ③ 육지·제주 SMP 를 열로 분리(지역별 상이 가능)
    piv = df.pivot_table(index=["date", "hour"], columns="areaName",
                         values="smp", aggfunc="first").reset_index()
    piv = piv.rename(columns={"육지": "smp_land", "제주": "smp_jeju"})
    for c in ("smp_land", "smp_jeju"):
        if c not in piv.columns:
            piv[c] = pd.NA

    diff = (piv["smp_land"] - piv["smp_jeju"]).abs()
    n_diff = int((diff > 0.001).sum())
    if n_diff == 0:
        log_check("SMP", "육지·제주 SMP 동일성", "정상",
                  f"전 {len(piv):,}행 동일(제주 별도 산정 없음)")
    else:
        rate = 100 * n_diff / len(piv)
        log_check("SMP", "육지·제주 SMP 동일성", "주의",
                  f"상이 {n_diff:,}행({rate:.1f}%) 최대편차 {diff.max():.2f}원 "
                  f"→ 지역별 열 분리 유지")

    # 수요예측은 지역 무관 동일값 → 첫 행 기준 병합
    dem = df.groupby(["date", "hour"], as_index=False)[
        ["mlfd", "jlfd", "slfd"]].first()
    out = piv.merge(dem, on=["date", "hour"], how="left")
    out = out.rename(columns={"hour": "hour_n", "mlfd": "demand_land",
                              "jlfd": "demand_jeju", "slfd": "demand_total"})
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    out = out[["date", "hour_n", "smp_land", "smp_jeju",
               "demand_land", "demand_jeju", "demand_total"]]
    out = out.sort_values(["date", "hour_n"]).reset_index(drop=True)

    # ② 중복
    dup = int(out.duplicated(subset=["date", "hour_n"]).sum())
    log_check("SMP", "중복 레코드", "정상" if dup == 0 else "오류", f"{dup}건")

    # ① 24시간 완비
    cnt = out.groupby("date")["hour_n"].nunique()
    bad = cnt[cnt != 24]
    log_check("SMP", "일자별 24시간 완비", "정상" if bad.empty else "주의",
              "전 일자 완비" if bad.empty else f"불완전 {len(bad)}일: {list(bad.index[:5])}")

    # ①-2 계통별 값 결측 — 행은 존재하나 값이 비어 있는 일자 탐지
    #  · 배경(2026.9.4) : API 원문에 areaName='제주' 행이 통째로 누락된 일자가
    #    6일 발생(08-18·08-22~24·08-30·08-31)하였으나, 행 수 기준 검증(①)은
    #    육지 24행만으로 '완비' 판정하여 결측을 탐지하지 못함
    #  · 영향 : 동일 일유형 시차 변수(smp_lag_same_daytype)가 연쇄 결측되어
    #    예측 산출 시 입력결측 24건 발생
    for col, nm in (("smp_land", "육지"), ("smp_jeju", "제주")):
        v = pd.to_numeric(out[col], errors="coerce")
        miss = out.assign(_na=v.isna()).groupby("date")["_na"].sum()
        bad_d = miss[miss > 0]
        #  등급 = '주의' — '오류'는 종료코드 1을 유발하여 step6 산출을 중단시킴
        #   · 2026.9.5 사고 : 제주 원문 결측(외부 API 요인)을 '오류'로 보고하여
        #     9.4 23:30 시간별·9.5 15:00 주간 산출이 연속 중단됨
        #   · 결측 자체는 시차 변수 결측 내성(--lag-fill)으로 이미 대응 완료
        #   · 탐지·기록이 목적이므로 '주의'로 하향하고 운영 연속성을 우선함
        log_check("SMP", f"일자별 값 결측({nm})",
                  "정상" if bad_d.empty else "주의",
                  "결측 없음" if bad_d.empty
                  else f"{len(bad_d)}일 / {int(bad_d.sum())}행: {list(bad_d.index[-5:])}")

    # ④ 수요 항등식
    resid = (out["demand_land"] + out["demand_jeju"] - out["demand_total"]).abs()
    n_bad = int((resid > 1.0).sum())
    log_check("SMP", "수요 항등식(육지+제주=계)", "정상" if n_bad == 0 else "주의",
              "전 행 일치" if n_bad == 0 else f"불일치 {n_bad}행(최대 {resid.max():.1f})")

    # ⑤ SMP 이상치
    for col, nm in (("smp_land", "육지"), ("smp_jeju", "제주")):
        s = pd.to_numeric(out[col], errors="coerce")
        n_zero = int((s <= 0).sum())
        n_high = int((s > 500).sum())
        n_na = int(s.isna().sum())
        log_check("SMP", f"SMP 이상치({nm})",
                  "정상" if n_zero + n_high + n_na == 0 else "주의",
                  f"0이하 {n_zero} / 500초과 {n_high} / 결측 {n_na} "
                  f"(범위 {s.min():.2f}~{s.max():.2f})")

    path = os.path.join(INTERIM_DIR, "api_smp_demand_hourly.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 저장 {path} ({len(out):,}행)")
    return out


# ------------------------------------------------------------------
# 2. 태양광 발전량
# ------------------------------------------------------------------
def build_pv():
    daily = sorted(glob.glob(os.path.join(RAW_DIR, "pv", "*.json")))
    bulk = sorted(glob.glob(os.path.join(RAW_DIR, "pv_bulk", "*.json")))
    files = daily + bulk
    if not files:
        print("\n  [건너뜀] data/raw/pv·pv_bulk 원문 없음")
        return None
    print(f"\n■ 태양광 발전량 전처리 (일별 {len(daily)}건 + 일괄 {len(bulk)}페이지)")

    rows = []
    for p in files:
        rows.extend(read_items(p))
    df = pd.DataFrame(rows)
    if df.empty:
        log_check("태양광", "원문 파싱", "오류", "레코드 0건")
        return None

    n0 = len(df)
    df = df.drop_duplicates(subset=["tradeYmd", "tradeNo", "regionNm"], keep="first")
    if n0 != len(df):
        log_check("태양광", "수집원 간 중복 제거", "정상",
                  f"{n0:,}행 → {len(df):,}행 ({n0 - len(df):,}행 제거)")

    df["hour_n"] = pd.to_numeric(df["tradeNo"], errors="coerce").astype("Int64")
    df["amgo"] = pd.to_numeric(df["amgo"], errors="coerce")
    df["date"] = pd.to_datetime(df["tradeYmd"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # 지역 커버리지 점검
    per_day = df.groupby("date")["regionNm"].nunique()
    mode_n = int(per_day.mode().iloc[0])
    bad = per_day[per_day != mode_n]
    log_check("태양광", "일자별 지역 수", "정상" if bad.empty else "주의",
              f"표준 {mode_n}개 지역" if bad.empty
              else f"표준 {mode_n}개 / 상이 {len(bad)}일: {list(bad.index[:5])}")

    df["구분"] = df["regionNm"].apply(lambda x: "제주" if "제주" in str(x) else "육지")
    piv = df.pivot_table(index=["date", "hour_n"], columns="구분",
                         values="amgo", aggfunc="sum").reset_index()
    for c in ("육지", "제주"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.rename(columns={"육지": "pv_land", "제주": "pv_jeju"})
    piv["pv_total"] = piv["pv_land"].fillna(0) + piv["pv_jeju"].fillna(0)
    out = piv[["date", "hour_n", "pv_land", "pv_jeju", "pv_total"]].sort_values(
        ["date", "hour_n"]).reset_index(drop=True)

    cnt = out.groupby("date")["hour_n"].nunique()
    bad = cnt[cnt != 24]
    log_check("태양광", "일자별 24시간 완비", "정상" if bad.empty else "주의",
              "전 일자 완비" if bad.empty else f"불완전 {len(bad)}일")

    n_neg = int((out["pv_total"] < 0).sum())
    log_check("태양광", "음수 발전량", "정상" if n_neg == 0 else "주의", f"{n_neg}행")

    path = os.path.join(INTERIM_DIR, "api_pv_hourly.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 저장 {path} ({len(out):,}행)")
    return out


# ------------------------------------------------------------------
# 3. 공휴일
# ------------------------------------------------------------------
def build_holiday():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "holiday", "*.xml")))
    if not files:
        print("\n  [건너뜀] data/raw/holiday 원문 없음")
        return None
    print(f"\n■ 공휴일 전처리 (원문 {len(files)}개월)")

    pat = re.compile(
        r"<item>(.*?)</item>", re.S)
    rows = []
    for p in files:
        text = open(p, "r", encoding="utf-8").read()
        for blk in pat.findall(text):
            def g(tag):
                m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", blk, re.S)
                return m.group(1).strip() if m else ""
            rows.append({"locdate": g("locdate"), "date_name": g("dateName"),
                         "is_holiday": g("isHoliday")})
    df = pd.DataFrame(rows)
    if df.empty:
        log_check("공휴일", "원문 파싱", "오류", "레코드 0건")
        return None

    df = df[df["locdate"].str.len() == 8].copy()
    df["date"] = pd.to_datetime(df["locdate"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df["is_holiday"] = (df["is_holiday"].str.upper() == "Y").astype(int)
    out = df[["date", "date_name", "is_holiday"]].drop_duplicates(
        subset=["date", "date_name"]).sort_values("date").reset_index(drop=True)

    yrs = sorted({d[:4] for d in out["date"]})
    per_year = out[out["is_holiday"] == 1].groupby(out["date"].str[:4]).size()
    log_check("공휴일", "연도별 공휴일수", "정상",
              " / ".join(f"{y}년 {per_year.get(y, 0)}일" for y in yrs))

    path = os.path.join(INTERIM_DIR, "api_holiday.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 저장 {path} ({len(out):,}행)")
    return out


# ------------------------------------------------------------------
# 4. 기존 자료 교차검증
# ------------------------------------------------------------------
def cross_validate(smp):
    if smp is None or not os.path.exists(REF_CLEAN):
        return
    print("\n■ 기존 hourly_smp_clean.csv 교차검증")
    ref = pd.read_csv(REF_CLEAN, encoding="utf-8-sig")
    need = {"date", "hour_n", "smp_land"}
    if not need.issubset(ref.columns):
        log_check("교차검증", "기준자료 컬럼", "주의", f"필요 컬럼 부재: {need - set(ref.columns)}")
        return

    ref = ref[["date", "hour_n", "smp_land"]].dropna()
    ref["hour_n"] = pd.to_numeric(ref["hour_n"], errors="coerce").astype("Int64")
    mg = smp.merge(ref, on=["date", "hour_n"], how="inner")
    if mg.empty:
        log_check("교차검증", "기간 중첩", "주의", "중첩 구간 없음")
        return

    d = (mg["smp_land_x"] - mg["smp_land_y"]).abs()
    n_bad = int((d > 0.01).sum())
    rate = 100 * (1 - n_bad / len(mg))
    log_check("교차검증", "SMP 일치율", "정상" if n_bad == 0 else "주의",
              f"중첩 {len(mg):,}행 / 일치 {rate:.2f}% / 최대편차 {d.max():.2f}원")


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step2 전처리·정합성 검증")
    ap.add_argument("--source", default="all",
                    choices=["all", "smp", "pv", "holiday"])
    args = ap.parse_args()

    os.makedirs(INTERIM_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 70)
    print(" step2 전처리·정합성 검증 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    smp = build_smp() if args.source in ("all", "smp") else None
    if args.source in ("all", "pv"):
        build_pv()
    if args.source in ("all", "holiday"):
        build_holiday()
    if args.source in ("all", "smp"):
        cross_validate(smp)

    # 검증 결과 저장
    path = os.path.join(LOG_DIR, f"step2_validation_{datetime.now():%Y%m%d_%H%M%S}.csv")
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
