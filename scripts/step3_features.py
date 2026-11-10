# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step3 피처 엔지니어링
================================================
목적 : 통합 데이터셋에 예측용 파생변수를 생성하고, 변수별 운영 가용성을 명시
입력 : data/processed/smp_dataset_hourly.csv (step3_build_dataset 산출)
출력 : data/processed/smp_features_hourly.csv  (학습·검증용 피처셋)
       data/processed/feature_dictionary.csv   (변수사전)
       logs/step3_features_report_{일시}.csv

설계 원칙
  · 예측 시점 : D-1 23시(익일 발전계획 확정 직후) 기준
  · 누설 방지 : 예측 시점에 확보 불가한 정보는 피처로 사용하지 않음
                (수요 실적·신재생 실적은 `operational=N`으로 표기하여 분석용으로만 보유)
  · 시차 변수 : lag24 이상만 사용(당일 값 참조 금지)

버전 : v1.8 (2026-10-04) — KPX 교재 반영: 일유형 플래그·경부하0원 레짐·경험가산분
       v1.7 (2026-09-23) — 명절 D−1(is_major_holiday_eve) · is_next/prev_day_off 달력 보완
       v1.6 (2026-09-01) — 강수·일사·태양광전망 운영 투입, 9.1 패턴 반영
       v1.5 (2026-09-01) — 급전순위 공개자료 연료비 변수 채택(--merit-feat)
       v1.4 (2026-08-29) — 연료비 구조 변수 결합(--fuel-feat) 신설
       v1.3 휴일 특화 변수 7종 추가
       v1.2 타깃별 피처셋 분리(제주 시차변수 오적용 수정)
       v1.1 시차 변수 정교화 13종 추가
       (동일 일유형 시차, 일중 형상 분리, 추세·비율, 수요–가격 결합)
실행 : python scripts\\step3_features.py
       python scripts\\step3_features.py --target smp_jeju
"""

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SRC = os.path.join(PROCESSED, "smp_dataset_hourly.csv")


def load_holiday_calendar():
    """공휴일 원천(data/raw/holiday/YYYYMM.xml)에서 전 구간 공휴일 일자 추출.

    목적 : 데이터셋 종료일 이후의 공휴일까지 포함하여 days_to_holiday 를 정확히 산출
           (데이터셋 행만 사용하면 예측 시점 기준행에서 상시 -1 이 산출됨)
    반환 : datetime64 목록. 원천 부재 시 빈 목록(기존 동작 유지)
    """
    return list(load_holiday_calendar_named()["dt"])


def load_holiday_calendar_named():
    """공휴일 원천에서 일자·명칭 추출 (명절 D−1 판별용).

    반환 : DataFrame[dt, date_name]. 원천 부재 시 빈 프레임
    """
    import glob
    import re
    import pandas as _pd

    hol_dir = os.path.join(BASE_DIR, "data", "raw", "holiday")
    rows = []
    if not os.path.isdir(hol_dir):
        return _pd.DataFrame(columns=["dt", "date_name"])
    for f in glob.glob(os.path.join(hol_dir, "*.xml")):
        try:
            with open(f, encoding="utf-8") as fh:
                txt = fh.read()
        except Exception:
            continue
        for m in re.finditer(r"<item>(.*?)</item>", txt, re.S):
            blk = m.group(1)
            if "<isHoliday>Y</isHoliday>" not in blk:
                continue
            d = re.search(r"<locdate>(\d{8})</locdate>", blk)
            if not d:
                continue
            nm = re.search(r"<dateName>([^<]*)</dateName>", blk)
            rows.append({
                "dt": _pd.to_datetime(d.group(1), format="%Y%m%d"),
                "date_name": (nm.group(1).strip() if nm else ""),
            })
    if not rows:
        return _pd.DataFrame(columns=["dt", "date_name"])
    return (_pd.DataFrame(rows)
            .drop_duplicates("dt")
            .sort_values("dt")
            .reset_index(drop=True))


def out_paths(target, variant=""):
    """타깃별 피처셋·변수사전 경로. 시차변수가 타깃 기준으로 생성되므로 분리 필수.

    variant : 실증용 변형 피처셋 접미사(예: "_fuel").
              운영 산출물(무접미사)을 덮어쓰지 않도록 분리 — A/B 비교 전제
    """
    return (os.path.join(PROCESSED, f"smp_features_{target}{variant}.csv"),
            os.path.join(PROCESSED, f"feature_dictionary_{target}{variant}.csv"))

DICT_ROWS = []
REPORT = []


def add_dict(name, 구분, 정의, 운영가용, 비고=""):
    DICT_ROWS.append({"변수명": name, "구분": 구분, "정의": 정의,
                      "운영가용": 운영가용, "비고": 비고})


def rep(항목, 값, 비고=""):
    REPORT.append({"항목": 항목, "값": str(값), "비고": 비고})
    print(f"  · {항목} : {값} {('- ' + 비고) if 비고 else ''}")


def cyc(df, col, period, name):
    """순환 인코딩(sin·cos) — 23시와 0시의 인접성 보존."""
    v = 2 * np.pi * df[col] / period
    df[f"{name}_sin"] = np.sin(v)
    df[f"{name}_cos"] = np.cos(v)
    add_dict(f"{name}_sin", "순환", f"{col} 사인 인코딩(주기 {period})", "Y")
    add_dict(f"{name}_cos", "순환", f"{col} 코사인 인코딩(주기 {period})", "Y")


def save_atomic(df, path):
    """임시파일 기록 후 원자적 교체.
    · 목적 : 저장 도중 중단(강제종료·전원차단) 시 기존 파일 손상 방지
    · 중단 시 임시파일만 잔존하며 기존 산출물은 무결성 유지"""
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)          # 동일 볼륨 내 원자적 교체
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def main():
    ap = argparse.ArgumentParser(description="step3 피처 엔지니어링")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"], help="시차변수 기준 타깃")
    ap.add_argument("--grp-feat", default="off", choices=["on", "off"],
                    help="가격결정발전기 그룹 시차 변수 사용 여부 "
                         "(기본 off — 2026.8.11 기각 판정, 개선 효과 없음)")
    ap.add_argument("--weather", default="off", choices=["off", "on"],
                    help="기상 변수 모델 투입 여부(기본 off — 2026.8.6 기각 판정 반영)")
    ap.add_argument("--fuel-feat", default="off", choices=["off", "on"],
                    help="연료비 구조 변수 투입 여부(내부 제원 기반 — 2026.8.31 기각)")
    ap.add_argument("--merit-feat", default="on", choices=["off", "on"],
                    help="급전순위 공개자료 기반 연료비 변수 투입(기본 사용) "
                         "— 2026.9.1 G-1 재판정 통과(D+1 -0.17%%p)")
    ap.add_argument("--markup-feat", default="off", choices=["off", "on"],
                    help="경험 가산분(SMP−가격결정기 변동비) 시차 변수 — "
                         "KPX 05번 NLP·SUP 대리(2026.10.4 신설) — 교육용 사본은 원천(사내자료) 미포함으로 기본 off")
    ap.add_argument("--d1-fill", default="off", choices=["off", "ffill", "land"],
                    help="전일값 파생(9종) 결측 대체 방식 — off·ffill·land(육지값, 제주 전용)")
    ap.add_argument("--lagw-fill", default="on", choices=["off", "on"],
                    help="절대시차(d2·w1) 결측 내성 — 동일 시간대 직전 유효값 대체. "
                         "2026.9.6 채택 — 기본 on. A/B 시 --lagw-fill off --tag _nolagw")
    ap.add_argument("--lag-fill", default="on", choices=["off", "on"],
                    help="동일 일유형·연휴위치 시차 변수의 결측 내성 적용 "
                         "(직전 '유효' 관측 참조). 2026.9.4 채택 — 기본 on. "
                         "A/B 비교 시 --lag-fill off --tag _nolagfill 사용")
    ap.add_argument("--netload", default="off", choices=["off", "on"],
                    help="순부하 변수군 투입(A-5, 2026.9.8 기각) — "
                         "net_load_fcst 및 파생 6종을 운영가용 Y로 분류. "
                         "A/B 시 --netload on --tag _netload")
    ap.add_argument("--tag", default="",
                    help="산출물 접미사(예: _nolagfill) — 실증용 분리 저장")
    args = ap.parse_args()
    VARIANT = "_fuel" if args.fuel_feat == "on" else ""
    VARIANT += args.tag              # 실증 시에만 지정 — 미지정 시 운영 경로
    OUT, DICT = out_paths(args.target, VARIANT)

    if not os.path.exists(SRC):
        sys.exit(f"[중단] 입력파일 없음 → {SRC}\n       step3_build_dataset.py 선행 실행 필요")

    print("=" * 70)
    print(" step3 피처 엔지니어링 (v1.8)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 시차 기준: {args.target}")
    print(f" 산출 구분: {'실증(연료비 결합)' if VARIANT else '운영(기준)'}")
    print("=" * 70)

    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour_n"] - 1, unit="h")
    df = df.sort_values("ts").reset_index(drop=True)

    print("\n■ ① 원본 확인")
    rep("행수", f"{len(df):,}행")
    rep("기간", f"{df['date'].min()} ~ {df['date'].max()}")
    gap = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
    rep("시간축 연속성", "정상" if len(gap) == len(df) else f"불일치({len(gap):,} vs {len(df):,})")

    # 기본 식별·타깃
    for c, g, d in (("date", "식별", "일자"), ("hour_n", "식별", "거래시간(1~24)"),
                    ("ts", "식별", "시각(구간 시작 기준)")):
        add_dict(c, g, d, "Y")
    add_dict("smp_land", "타깃", "육지 계통한계가격(원/kWh)", "-", "예측 대상")
    add_dict("smp_jeju", "타깃", "제주 계통한계가격(원/kWh)", "-", "예측 대상")

    # ── ② 달력·순환 변수 ───────────────────────
    print("\n■ ② 달력·순환 변수")
    df["doy"] = df["ts"].dt.dayofyear
    df["quarter"] = df["ts"].dt.quarter
    df["season"] = df["month"].map(
        {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
         6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"})
    df["is_peak_season"] = df["month"].isin([1, 2, 7, 8, 12]).astype(int)
    for c, g, d in (("year", "달력", "연도"), ("month", "달력", "월"),
                    ("day", "달력", "일"), ("dow", "달력", "요일(0=월)"),
                    ("week_of_year", "달력", "연중 주차"), ("doy", "달력", "연중 일수"),
                    ("quarter", "달력", "분기"), ("season", "달력", "계절"),
                    ("is_peak_season", "달력", "동·하계 첨두기(1·2·7·8·12월)"),
                    ("is_weekend", "달력", "주말 여부"),
                    ("is_holiday", "달력", "공휴일 여부"),
                    ("daytype", "달력", "평일·휴일 구분")):
        add_dict(c, g, d, "Y")

    cyc(df, "hour_n", 24, "hour")
    cyc(df, "dow", 7, "dow")
    cyc(df, "month", 12, "month")
    cyc(df, "doy", 365.25, "doy")

    # 연휴 구조
    #  · 익일·전일 휴일 여부: 데이터셋 끝단에서 shift 가 미래 연휴를 못 보면
    #    is_next_day_off=0 으로 남는 결함 → 공휴일 XML·주말 달력으로 보완
    daily = df.drop_duplicates("date")[["date", "daytype"]].reset_index(drop=True)
    daily["dt"] = pd.to_datetime(daily["date"])
    daily["off"] = (daily["daytype"] == "휴일").astype(int)
    off_by_dt = dict(zip(daily["dt"], daily["off"]))
    hol_set = set(pd.to_datetime(load_holiday_calendar()))

    def _is_off_day(ts: pd.Timestamp) -> int:
        ts = pd.Timestamp(ts).normalize()
        if ts in off_by_dt:
            return int(off_by_dt[ts])
        return int((ts.dayofweek >= 5) or (ts in hol_set))

    daily["prev_off"] = (daily["dt"] - pd.Timedelta(days=1)).map(_is_off_day).astype(int)
    daily["next_off"] = (daily["dt"] + pd.Timedelta(days=1)).map(_is_off_day).astype(int)
    grp = (daily["off"] != daily["off"].shift()).cumsum()
    daily["run_len"] = daily.groupby(grp)["off"].transform("size") * daily["off"]
    df = df.merge(daily[["date", "prev_off", "next_off", "run_len"]], on="date", how="left")
    df = df.rename(columns={"prev_off": "is_prev_day_off",
                            "next_off": "is_next_day_off",
                            "run_len": "holiday_run_len"})
    add_dict("is_prev_day_off", "달력", "전일 휴일 여부", "Y")
    add_dict("is_next_day_off", "달력", "익일 휴일 여부(미래 공휴일·주말 포함)", "Y")
    add_dict("holiday_run_len", "달력", "연휴 연속일수(평일은 0)", "Y")

    # ── 휴일 특화 변수(v1.3) ───────────────────
    # 배경 : 휴일 MAPE가 평일의 2.1배로 최대 취약 구간 → 휴일 구조 정보 보강
    d2 = df.drop_duplicates("date")[["date", "daytype", "is_holiday",
                                     "holiday_name"]].reset_index(drop=True)
    d2["dt"] = pd.to_datetime(d2["date"])
    d2["off"] = (d2["daytype"] == "휴일").astype(int)

    # (1) 공휴일 기준 경과·잔여 일수
    #  【결함 조치 2026.8.17】
    #   기존 : 데이터셋에 존재하는 행에서만 공휴일을 추출
    #          → 데이터셋 종료일(예측 시 D+1) 이후 공휴일이 보이지 않아
    #            days_to_holiday 가 -1(미상)로 산출됨
    #          → ① 운영 예측 시 상시 오값 투입(train-serving skew)
    #             ② 데이터 축적 후 재산출 시 값이 바뀌어 재현성 훼손
    #   조치 : 공휴일 원천(data/raw/holiday/*.xml)을 직접 읽어 **미래 구간 포함**
    #          달력을 구성하고, 데이터셋 추출분과 합집합으로 사용
    hol_dt = pd.concat([
        d2.loc[d2["is_holiday"] == 1, "dt"],
        pd.Series(load_holiday_calendar(), dtype="datetime64[ns]"),
    ]).drop_duplicates().sort_values()
    prev_h = pd.Series(pd.NaT, index=d2.index, dtype="datetime64[ns]")
    next_h = pd.Series(pd.NaT, index=d2.index, dtype="datetime64[ns]")
    if not hol_dt.empty:
        arr = hol_dt.sort_values().to_numpy()
        idx = np.searchsorted(arr, d2["dt"].to_numpy(), side="right")
        prev_h = pd.Series([arr[i - 1] if i > 0 else pd.NaT for i in idx], index=d2.index)
        idx2 = np.searchsorted(arr, d2["dt"].to_numpy(), side="left")
        next_h = pd.Series([arr[i] if i < len(arr) else pd.NaT for i in idx2], index=d2.index)
    d2["days_since_holiday"] = (d2["dt"] - pd.to_datetime(prev_h)).dt.days.fillna(-1).astype(int)
    d2["days_to_holiday"] = (pd.to_datetime(next_h) - d2["dt"]).dt.days.fillna(-1).astype(int)

    # (2) 연휴 내 위치 : 0=평일 1=단일 2=첫날 3=중간 4=마지막
    blk = (d2["off"] != d2["off"].shift()).cumsum()
    pos = d2.groupby(blk).cumcount() + 1
    size = d2.groupby(blk)["off"].transform("size")
    d2["holiday_position"] = np.select(
        [d2["off"] == 0, size == 1, pos == 1, pos == size],
        [0, 1, 2, 4], default=3).astype(int)

    # (3) 샌드위치 평일(앞뒤 모두 휴일인 평일)
    d2["is_bridge_day"] = ((d2["off"] == 0)
                           & (d2["off"].shift(1) == 1)
                           & (d2["off"].shift(-1) == 1)).astype(int)

    # (4) 명절 여부(설·추석 연휴)
    nm = d2["holiday_name"].fillna("").astype(str)
    major = nm.str.contains("설날|추석")
    d2["is_major_holiday"] = major.astype(int)
    # 명절 연휴 전체(전후일 포함) 확산 — 데이터셋 내 연속 휴일 블록
    mblk = blk.where(d2["off"] == 1)
    major_blocks = set(mblk[major].dropna().unique())
    d2["is_major_holiday_period"] = mblk.isin(major_blocks).fillna(False).astype(int)

    # (5) 명절 연휴 바로 전날(D−1)
    #  · 데이터셋 종료일 때문에 shift(-1)만 쓰면 예측 직전일이 eve=0으로 남는 결함
    #    → 공휴일 원천 XML의 명절 연휴 블록을 기준으로 판정
    #  · 9.23 사례: 법정휴일 아님 + 익일(9.24)이 추석 연휴 → eve=1
    hol_named = load_holiday_calendar_named()
    major_period_dates: set = set()
    if not hol_named.empty:
        hol_named = hol_named.sort_values("dt").reset_index(drop=True)
        hol_named["is_major"] = (hol_named["date_name"].fillna("")
                                 .astype(str).str.contains("설날|추석"))
        # 연속 공휴일 블록 중 설·추석이 포함된 블록 전체
        gap = hol_named["dt"].diff().dt.days.fillna(2)
        hol_named["hblk"] = (gap > 1).cumsum()
        maj_blks = set(hol_named.loc[hol_named["is_major"], "hblk"].unique())
        major_period_dates = set(
            hol_named.loc[hol_named["hblk"].isin(maj_blks), "dt"].tolist())
    # 데이터셋에 이미 표시된 명절 연휴일도 합집합
    if "is_major_holiday_period" in d2.columns:
        major_period_dates |= set(
            d2.loc[d2["is_major_holiday_period"] == 1, "dt"].tolist())
    tomorrow = d2["dt"] + pd.Timedelta(days=1)
    d2["is_major_holiday_eve"] = (
        (d2["off"] == 0) & tomorrow.isin(major_period_dates)
    ).astype(int)

    df = df.merge(d2[["date", "days_since_holiday", "days_to_holiday",
                      "holiday_position", "is_bridge_day",
                      "is_major_holiday", "is_major_holiday_period",
                      "is_major_holiday_eve"]],
                  on="date", how="left")
    add_dict("days_since_holiday", "휴일", "직전 공휴일로부터 경과일수", "Y")
    add_dict("days_to_holiday", "휴일", "차기 공휴일까지 잔여일수", "Y")
    add_dict("holiday_position", "휴일",
             "연휴 내 위치(0평일·1단일·2첫날·3중간·4마지막)", "Y")
    add_dict("is_bridge_day", "휴일", "샌드위치 평일 여부", "Y")
    add_dict("is_major_holiday", "휴일", "명절 당일 여부(설·추석)", "Y")
    add_dict("is_major_holiday_period", "휴일", "명절 연휴 기간 여부", "Y")
    add_dict("is_major_holiday_eve", "휴일",
             "명절 연휴 바로 전날(D−1) 여부 — 연차·조업축소 수요↓", "Y")

    # ── ②-b KPX 준용 일유형·경부하 0원 레짐 (2026.10.4, 교재 05·07) ──
    #  · 07번 54·55쪽 : 명절±2일·징검다리·연휴 전후를 평일과 분리 예측
    #  · 05번 붙임1 : 특수경부하 시 시장가격 0원 — 주간·휴일·고재생 동시 조건
    print("\n■ ②-b KPX 일유형·경부하 레짐")
    from kpx_daytype import attach_kpx_flags
    df = attach_kpx_flags(df)
    for c, dfn in (
        ("is_kpx_myeong_adj", "명절 전후(±2일) 근무일 — KPX 07번 준용"),
        ("is_kpx_bridge", "징검다리 평일 — KPX 07번 준용"),
        ("is_kpx_holiday_eve", "연휴 전일(연속휴일≥3일 직전) — KPX 07번"),
        ("is_kpx_holiday_return", "연휴 익일(연속휴일≥3일 직후) — KPX 07번"),
        ("is_kpx_special_weekday", "특수근무일(명절전후·징검다리) — 편의창 제외 대상"),
    ):
        add_dict(c, "휴일", dfn, "Y", "2026.10.4 KPX 교재 채택")
    # 경부하·0원 레짐 — 시계·달력만 사용(순부하 기각 회피)
    df["is_midday"] = df["hour_n"].between(11, 16).astype("int8")
    maj_per = (df["is_major_holiday_period"].fillna(0).astype(int)
               if "is_major_holiday_period" in df.columns
               else pd.Series(0, index=df.index))
    light_day = (
        (df["is_holiday"].fillna(0).astype(int) == 1)
        | (df["is_weekend"].fillna(0).astype(int) == 1)
        | (maj_per == 1)
        | (df["is_kpx_special_weekday"] == 1)
    )
    df["is_light_load_regime"] = (
        (df["is_midday"] == 1) & light_day).astype("int8")
    add_dict("is_midday", "달력", "주간 한낮(11~16시) — 덕커브·0원 취약 구간", "Y",
             "2026.10.4 KPX 05번·관찰(9.25~27) 반영")
    add_dict("is_light_load_regime", "휴일",
             "경부하 0원 위험 레짐 = 한낮 × (휴일·주말·명절·특수근무일)", "Y",
             "2026.10.4 — 05번 특수경부하 0원 메커니즘 대리")
    if "kpx_class" in df.columns:
        add_dict("kpx_class", "휴일", "KPX 준용 일유형 문자열(분석용)", "N",
                 "모델 직접 투입 제외 — 이진 플래그로 투입")
    rep("생성 변수",
        f"KPX 플래그 5종 + 레짐 2종 / 특수근무일 "
        f"{int(df.drop_duplicates('date')['is_kpx_special_weekday'].sum())}일")
    rep("생성 변수", "달력·순환 20종 + 휴일 특화 7종 + KPX 레짐")

    # ── ③ 수요예측 파생 ────────────────────────
    print("\n■ ③ 수요예측 파생변수")
    g = df.groupby("date")["demand_fcst_total"]
    df["demand_fcst_daily_mean"] = g.transform("mean")
    df["demand_fcst_daily_max"] = g.transform("max")
    df["demand_fcst_daily_min"] = g.transform("min")
    df["demand_fcst_ratio"] = df["demand_fcst_total"] / df["demand_fcst_daily_mean"]
    # 순위 산출 — 수요예측 결측 구간(미공표 익일 등)은 0으로 대체 후 정수화
    #  · 예측 대상일 행이 데이터셋에 포함되면서 결측이 발생할 수 있음(v1.1, 2026.8.10)
    df["demand_fcst_rank"] = g.rank(method="first").fillna(0).astype(int)
    df["demand_fcst_range"] = df["demand_fcst_daily_max"] - df["demand_fcst_daily_min"]
    df["demand_fcst_jeju_share"] = df["demand_fcst_jeju"] / df["demand_fcst_total"]
    for c, d in (("demand_fcst_land", "육지 수요예측(MW)"),
                 ("demand_fcst_jeju", "제주 수요예측(MW)"),
                 ("demand_fcst_total", "계통 수요예측(MW)"),
                 ("demand_fcst_daily_mean", "당일 수요예측 일평균"),
                 ("demand_fcst_daily_max", "당일 수요예측 최대"),
                 ("demand_fcst_daily_min", "당일 수요예측 최소"),
                 ("demand_fcst_ratio", "일평균 대비 시간별 수요 배율"),
                 ("demand_fcst_rank", "일중 수요 순위(1~24)"),
                 ("demand_fcst_range", "일중 수요 진폭"),
                 ("demand_fcst_jeju_share", "제주 수요 비중")):
        add_dict(c, "수요예측", d, "Y", "D-1 23시 공표")
    rep("생성 변수", "수요예측 파생 7종")

    # ── ④ 시차·이동통계 (타깃 기준) ────────────
    print(f"\n■ ④ 시차·이동통계 ({args.target} 기준)")
    t = args.target
    #  --lagw-fill on : 절대시차 변수의 결측을 동일 시간대 직전 유효값으로 대체
    #   · 배경(2026.9.6) : 제주 SMP 원문 부재 6일(8.18·8.22~24·8.30·8.31)이
    #     7일 시차 변수로 전파되어 대상일 9.6. 입력결측 72건 발생
    #     (smp_lag_w1 · smp_trend_d1_w1 · smp_ratio_d1_w1 = 3변수 x 24시간)
    #   · 원문 재수집 불가 확인 — API 응답 자체에 제주 행이 없음(기수집 14건 전량)
    #   · **smp_lag_d1 은 제외** — naive-1 벤치마크 정의이므로 변경 금지
    #   · 누출 없음 — shift 선행으로 과거값만 사용, ffill 은 과거값을 전방 전파
    #   · 채택 근거(2026.9.6 A/B, 동일 홀드아웃)
    #     제주 : MAE 10.48 → 10.36원 / rMAE 0.803 → 0.794 / 편의 5.46 → 5.30%
    #            성분별 개선 최대는 diff(11.97 → 11.55원)
    #     육지 : 8.29원 / 0.732 — 결측 없어 **완전 동일**(무영향 확인)
    #   · 입력결측 72건 → 0건 — 산출 안정성이 본 조치의 주목적
    for lag, nm in ((24, "d1"), (48, "d2"), (168, "w1")):
        col = f"smp_lag_{nm}"
        df[col] = df[t].shift(lag)
        if args.lagw_fill == "on" and nm != "d1":
            df[col] = df.groupby("hour_n")[col].ffill()
        add_dict(col, "시차", f"{lag}시간 전 {t}", "Y",
                 "예측 시점 기준 확보 가능"
                 + (" / 결측 내성 적용" if (args.lagw_fill == "on" and nm != "d1")
                    else ""))
    #  파생 산출용 전일값 — smp_lag_d1 원본은 naive-1 벤치마크이므로 보존하고,
    #  파생 변수만 결측 내성 적용본(d1f)으로 산출
    #   · 배경(2026.9.7) : 제주 SMP 원문 부재가 일·월요일에 반복 발생(2026.8.18 이후 8일)
    #     대상일 전일이 결측이면 smp_lag_d1 및 파생 9종이 전량 결측 → 입력결측 288건
    #   · 조치 : 파생 9종을 d1f 기반으로 산출하여 결측을 1변수(24건)로 축소
    #   · 대체 방식(--d1-fill)
    #     off   : 미적용(원본 유지)
    #     ffill : 동일 시간대 직전 유효값 — 일요일 결측 시 토요일값이 월요일에 전파되어
    #             일유형 불일치 위험. 2026.9.7 A/B에서 MAE 10.36 → 10.46원 악화
    #     land  : **동일 시점 육지 SMP로 대체**(제주 전용) — 최근 육지·제주 일치율
    #             2026.6월 99.86% / 7월 100% / 8월 97.67%로 근접
    if args.d1_fill == "land" and t == "smp_jeju" and "smp_land" in df.columns:
        d1f = df["smp_lag_d1"].fillna(df["smp_land"].shift(24))
    elif args.d1_fill == "ffill":
        d1f = df.groupby("hour_n")["smp_lag_d1"].ffill()
    else:
        d1f = df["smp_lag_d1"]
    df["smp_lag_d1_dmean"] = d1f.groupby(df["date"]).transform("mean")
    df["smp_lag_d1_dmax"] = d1f.groupby(df["date"]).transform("max")
    df["smp_lag_d1_dmin"] = d1f.groupby(df["date"]).transform("min")
    add_dict("smp_lag_d1_dmean", "시차", "전일 SMP 일평균", "Y")
    add_dict("smp_lag_d1_dmax", "시차", "전일 SMP 최대", "Y")
    add_dict("smp_lag_d1_dmin", "시차", "전일 SMP 최소", "Y")

    # 동시간대 이동평균(전일 기준 시점부터 소급)
    s = df[t].shift(24)
    for w, nm in ((7 * 24, "ma7"), (30 * 24, "ma30")):
        col = f"smp_{nm}"
        df[col] = s.rolling(w, min_periods=max(24, w // 4)).mean()
        add_dict(col, "시차", f"직전 {w // 24}일 SMP 이동평균(lag24 기준)", "Y")
    df["smp_same_hour_ma7"] = (
        df.groupby("hour_n")[t].transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean()))
    add_dict("smp_same_hour_ma7", "시차", "동일 시간대 직전 7일 평균", "Y")
    df["smp_trend_d1_w1"] = d1f - df["smp_lag_w1"]
    add_dict("smp_trend_d1_w1", "시차", "전일 대비 전주 동시간 SMP 차", "Y")

    # ── 시차 정교화(v1.1) ──────────────────────
    # (1) 동일 일유형 직전일 동시간 — 벤치마크 B5를 변수로 편입
    #  --lag-fill on(기본) : shift(1) 결과가 결측이면 직전 '유효' 관측으로 대체
    #   · 채택 근거(2026.9.4 A/B, D+1 홀드아웃 동일 표본)
    #     기준 MAE 8.98 / rMAE 0.804 → 적용 8.89 / 0.795 (MAE -0.09원)
    #   · 배경 : API 원문에 제주 행이 누락된 일자(2026 6일)가 직전 동일 일유형일 때
    #     시차 변수가 24시간 전량 결측되어 예측 산출이 저해됨
    #   · 누출 없음 — shift(1) 선행으로 과거값만 사용, ffill은 과거값을 전방 전파
    #   · 변수 정의('직전 발생 동일 일유형')와도 정합
    def _lag(g_key):
        s0 = df.groupby(g_key)[t].shift(1)
        if args.lag_fill != "on":
            return s0
        return s0.groupby(g_key).ffill()

    key = df["daytype"].astype(str) + "_" + df["hour_n"].astype(str)
    df["smp_lag_same_daytype"] = _lag(key)
    add_dict("smp_lag_same_daytype", "시차",
             "동일 일유형(평일·휴일)의 직전 일자 동시간 SMP", "Y",
             "최우수 벤치마크(B5) 규칙의 변수화"
             + (" / 결측 내성 적용" if args.lag_fill == "on" else " / 결측 내성 미적용"))

    # (1-2) 동일 연휴위치 직전값 — 명절·연휴 패턴 반복성 포착
    key2 = df["holiday_position"].astype(str) + "_" + df["hour_n"].astype(str)
    df["smp_lag_same_position"] = _lag(key2)
    add_dict("smp_lag_same_position", "휴일",
             "동일 연휴위치·동시간의 직전 발생 SMP", "Y")

    # (2) 전일 일중 형상 — 수준과 형상 분리
    df["smp_resid_d1"] = d1f - df["smp_lag_d1_dmean"]
    df["smp_shape_d1"] = d1f / df["smp_lag_d1_dmean"].replace(0, np.nan)
    df["smp_range_d1"] = df["smp_lag_d1_dmax"] - df["smp_lag_d1_dmin"]
    df["smp_std_d1"] = d1f.groupby(df["date"]).transform("std")
    add_dict("smp_resid_d1", "시차", "전일 동시간 − 전일 일평균(일중 편차)", "Y")
    add_dict("smp_shape_d1", "시차", "전일 동시간 ÷ 전일 일평균(일중 형상비)", "Y")
    add_dict("smp_range_d1", "시차", "전일 SMP 일중 진폭", "Y")
    add_dict("smp_std_d1", "시차", "전일 SMP 일중 표준편차", "Y")

    # (3) 추세·비율
    df["smp_ratio_d1_w1"] = d1f / df["smp_lag_w1"].replace(0, np.nan)
    df["smp_diff_d1_d2"] = d1f - df["smp_lag_d2"]
    df["smp_ewm7"] = s.ewm(span=7 * 24, min_periods=48).mean()
    add_dict("smp_ratio_d1_w1", "시차", "전일 ÷ 전주 동시간 SMP", "Y")
    add_dict("smp_diff_d1_d2", "시차", "전일 − 전전일 동시간 SMP(단기 추세)", "Y")
    add_dict("smp_ewm7", "시차", "지수가중이동평균(span 7일, lag24 기준)", "Y")

    # (4) 수요–가격 결합
    dfc = df["demand_fcst_total"]
    df["demand_fcst_diff_d1"] = dfc - dfc.shift(24)
    df["demand_fcst_ratio_d1"] = dfc / dfc.shift(24).replace(0, np.nan)
    df["smp_per_demand_d1"] = d1f / dfc.shift(24).replace(0, np.nan) * 1000
    add_dict("demand_fcst_diff_d1", "결합", "금일 − 전일 동시간 수요예측(MW)", "Y")
    add_dict("demand_fcst_ratio_d1", "결합", "금일 ÷ 전일 동시간 수요예측", "Y")
    add_dict("smp_per_demand_d1", "결합",
             "전일 동시간 SMP ÷ 전일 수요예측(×1000) — 단위수요당 가격", "Y")

    rep("생성 변수", "시차·이동통계 10종 + 정교화 13종")
    rep("초기 결측", f"{int(df['smp_ma30'].isna().sum()):,}행",
        "이동평균 워밍업 구간 — 학습 시 제외 권장")

    # ── ④-2 기상 변수 ──────────────────────────
    # ── ④-1 가격결정발전기(그룹) 시차 변수 ─────
    #  · R-1 수령(2026.8.11)으로 2021~2022 구간 확보 → 보유율 58.0% → 98.0%
    #  · grp 자체는 사후 확정 정보(operational=N)이나, lag24 이상은 예측 시점 확보 가능
    #  · 한계발전기 연료원이 SMP 수준을 직접 결정하므로 설명력 기대
    if args.grp_feat == "on" and "grp" in df.columns and df["grp"].notna().any():
        print("\n■ ④-1 가격결정발전기 그룹 시차 변수")
        gs = df["grp"].astype(str).replace("nan", np.nan)
        # ① 직전일·직전주 동시간 결정 그룹(범주형)
        for lag, nm in ((24, "d1"), (168, "w1")):
            col = f"grp_lag_{nm}"
            df[col] = gs.shift(lag).astype("category")
            add_dict(col, "시차", f"{lag}시간 전 가격결정 그룹", "Y",
                     "예측 시점 확보 가능 — 실적 확정분")
        # ② 연료원 구분 — 석탄 결정 여부(저가 구간 신호)
        df["is_coal_lag_d1"] = (gs.shift(24) == "석탄").astype(int)
        add_dict("is_coal_lag_d1", "시차", "직전일 동시간 석탄 결정 여부", "Y",
                 "석탄 한계 시 SMP 저가 구간")
        # ③ 직전 7일간 석탄 결정 비중 — 급전 여건 대리지표
        df["coal_ratio_w1"] = (gs.shift(24) == "석탄").rolling(168, min_periods=24).mean()
        add_dict("coal_ratio_w1", "시차", "직전 7일 석탄 결정 비중(lag24 기준)", "Y",
                 "급전 여건 대리지표")
        # ④ 그룹 전환 여부 — 한계발전기 교체 신호
        df["grp_change_d1"] = (gs.shift(24) != gs.shift(48)).astype(int)
        add_dict("grp_change_d1", "시차", "직전일 대비 결정 그룹 변경 여부", "Y")
        rep("그룹 시차 변수", "5종 생성",
            f"grp 보유율 {df['grp'].notna().mean() * 100:.1f}%")

    print("\n■ ④-2 기상 변수")
    WBASE = [c for c in ("temp_land", "temp_jeju", "humid_land", "humid_jeju",
                         "wind_spd_land", "solar_land", "solar_jeju",
                         "sunshine_land", "rain_land") if c in df.columns]
    if not WBASE:
        rep("기상 변수", "미보유 — 생성 생략",
            "step1_collect_weather.py → step2_parse_weather.py 선행 시 반영")
    else:
        # 운영가용 표기 — off 시 생성은 하되 모델 입력에서 제외(분석 전용)
        #  · 근거 : 2026.8.6 검증 결과 육지·제주 모두 개선 0.00%p
        #           하이퍼파라미터 재탐색 후에도 동일, 변수 중요도 상위 15 미진입
        #           KPX 수요예측값에 기상 전망이 내재되어 정보 중복
        # 2026.9.1 사례 반영 — 강수·일사·일조는 순부하 경로로 운영 투입
        # 기온·습도·풍속 계열은 수요예측과 중복되어 종전 기각 유지
        # 2026.9.1 재분류 — 기상청 단기예보(VilageFcst) 제공 항목만 운영가용
        #  · 제공     : 기온(TMP)·습도(REH)·풍속(WSD)·**강수(PCP)**
        #  · 미제공   : **일사·일조** → 운영 시 확보 불가하므로 미가용(N) 처리
        #  · 성능 영향: 일사·일조 제외 시 D+1 8.42% → 8.41%로 **차이 없음**(실측 확인)
        WX_RE = {"rain_land"}
        WX_SOLAR = {"solar_land", "solar_jeju", "sunshine_land"}
        WNOTE_RE = ("2026.9.1 채택 — 재생불리(강수) 경로. "
                    "기상청 단기예보 PCP로 D+1 확보 가능")
        WNOTE_SOLAR = ("2026.9.1 분석 전용 — 기상청 단기예보 일사·일조 미제공으로 "
                       "운영 확보 불가. 제외 시 성능 차이 없음(8.42→8.41%)")
        WNOTE_OLD = ("기각(2026.8.6) — 수요예측과 정보 중복. 분석·해석 전용")

        def wop(col):
            if col in WX_RE:
                return "Y", WNOTE_RE
            if col in WX_SOLAR:
                return "N", WNOTE_SOLAR
            if args.weather == "on":
                return "Y", "운영 시 기상청 단기예보값으로 대체 투입"
            return "N", WNOTE_OLD

        # 계통별 대표 기온 — 제주 관측 결측 시 육지값으로 대체
        tcol = "temp_land" if "temp_land" in df.columns else None
        tj = "temp_jeju" if "temp_jeju" in df.columns else tcol

        for c in WBASE:
            op, nt = wop(c)
            add_dict(c, "기상", {
                "temp_land": "육지 권역 가중평균 기온(℃)",
                "temp_jeju": "제주 기온(℃)",
                "humid_land": "육지 상대습도(%)", "humid_jeju": "제주 상대습도(%)",
                "wind_spd_land": "육지 풍속(m/s)",
                "solar_land": "육지 일사량(MJ/m²)", "solar_jeju": "제주 일사량(MJ/m²)",
                "sunshine_land": "육지 일조시간(hr)", "rain_land": "육지 강수량(mm)",
            }.get(c, c), op, nt)

        # ① 냉난방 도시(degree hours) — 수요와의 비선형 관계 반영
        for pre, col in (("land", tcol), ("jeju", tj)):
            if not col:
                continue
            df[f"hdd_{pre}"] = (18.0 - df[col]).clip(lower=0)
            df[f"cdd_{pre}"] = (df[col] - 24.0).clip(lower=0)
            add_dict(f"hdd_{pre}", "기상", f"{pre} 난방도시 = max(0, 18℃ - 기온)", "N", WNOTE_OLD)
            add_dict(f"cdd_{pre}", "기상", f"{pre} 냉방도시 = max(0, 기온 - 24℃)", "N", WNOTE_OLD)

        # ② 불쾌지수 — 기온·습도 결합(체감 냉방수요 대리변수)
        if tcol and "humid_land" in df.columns:
            df["thi_land"] = (1.8 * df[tcol] - 0.55 * (1 - df["humid_land"] / 100)
                              * (1.8 * df[tcol] - 26) + 32)
            add_dict("thi_land", "기상", "육지 불쾌지수(THI)", "N", WNOTE_OLD)

        # ③ 일평균·일교차·누적일사 — 일 단위 집약
        if tcol:
            g = df.groupby("date")[tcol]
            df["temp_daily_mean"] = g.transform("mean")
            df["temp_daily_max"] = g.transform("max")
            df["temp_daily_min"] = g.transform("min")
            df["temp_daily_range"] = df["temp_daily_max"] - df["temp_daily_min"]
            for c, d in (("temp_daily_mean", "당일 평균기온(℃)"),
                         ("temp_daily_max", "당일 최고기온(℃)"),
                         ("temp_daily_min", "당일 최저기온(℃)"),
                         ("temp_daily_range", "당일 일교차(℃)")):
                add_dict(c, "기상", d, "N", WNOTE_OLD)
        if "solar_land" in df.columns:
            df["solar_cum_day"] = df.groupby("date")["solar_land"].cumsum()
            add_dict("solar_cum_day", "기상", "당일 누적 일사량(MJ/m²)", "N", WNOTE_SOLAR)

        # ④ 이동평균·시차 — 축열 효과 및 전일 대비 변화
        if tcol:
            s = df[tcol]
            df["temp_ma24"] = s.rolling(24, min_periods=12).mean()
            df["temp_ma72"] = s.rolling(72, min_periods=36).mean()
            df["temp_lag24"] = s.shift(24)
            df["temp_diff_d1"] = s - df["temp_lag24"]
            for c, d, n in (("temp_ma24", "기온 24시간 이동평균(℃)", ""),
                            ("temp_ma72", "기온 72시간 이동평균(℃)", "축열 효과 반영"),
                            ("temp_lag24", "전일 동시간 기온(℃)", "운영 시점 확정 가용"),
                            ("temp_diff_d1", "금일 - 전일 동시간 기온차(℃)", "")):
                add_dict(c, "기상", d, "N", (n + " / " + WNOTE_OLD) if n else WNOTE_OLD)

        # ⑤ 극한기상 지시변수 — 수요 급증 구간 식별
        if tcol:
            df["is_heatwave"] = (df["temp_daily_max"] >= 33).astype("Int8")
            df["is_coldwave"] = (df["temp_daily_min"] <= -12).astype("Int8")
            add_dict("is_heatwave", "기상", "폭염일(당일 최고기온 33℃ 이상)", "N", WNOTE_OLD)
            add_dict("is_coldwave", "기상", "한파일(당일 최저기온 -12℃ 이하)", "N", WNOTE_OLD)

        made = [c for c in df.columns if c.startswith(("hdd_", "cdd_", "thi_",
                "temp_daily", "temp_ma", "temp_lag", "temp_diff", "solar_cum",
                "is_heat", "is_cold"))]
        rep("생성 변수", f"기상 기본 {len(WBASE)}종 + 파생 {len(made)}종")
        rep("기상 적용률", f"{df[WBASE[0]].notna().mean() * 100:.1f}%")
        rep("모델 투입", "강수·일사·일조 운영 / 기온·습도·풍속 분석전용", WNOTE_RE)

        # ── ④-2b 재생불리 운영변수 (2026.9.1) ─────────
        print("\n■ ④-2b 재생불리 운영변수")
        day = df["hour_n"].between(9, 17)
        rain_dt = (df.loc[day].groupby("date")["rain_land"].sum()
                   if "rain_land" in df.columns else pd.Series(dtype=float))
        solar_dt = (df.loc[day].groupby("date")["solar_land"].sum()
                    if "solar_land" in df.columns else pd.Series(dtype=float))
        df = df.merge(rain_dt.rename("rain_daytime"), left_on="date",
                      right_index=True, how="left")
        df = df.merge(solar_dt.rename("solar_daytime"), left_on="date",
                      right_index=True, how="left")
        df["is_rain_day"] = (df["rain_daytime"].fillna(0) >= 1.0).astype("int8")
        # 일사 기후값 — 직전 60일의 동월 주간합 중앙값 (당일 누설 방지)
        tmp = df.drop_duplicates("date")[["date", "month", "solar_daytime"]].copy()
        tmp = tmp.sort_values("date")
        tmp["solar_climo"] = (tmp.groupby("month")["solar_daytime"]
                              .transform(lambda s: s.shift(1).rolling(60, min_periods=10).median()))
        df = df.merge(tmp[["date", "solar_climo"]], on="date", how="left")
        df["solar_ratio"] = df["solar_daytime"] / df["solar_climo"].replace(0, np.nan)
        # 예보 미달일(주간 실측/예보 5시간 미만)은 저일사로 오인하지 않음
        cov = (df.loc[day].groupby("date")["solar_land"]
               .apply(lambda s: int(s.notna().sum())))
        df = df.merge(cov.rename("solar_day_cov"), left_on="date",
                      right_index=True, how="left")
        df.loc[df["solar_day_cov"].fillna(0) < 5, "solar_ratio"] = np.nan
        df["solar_deficit"] = (1.0 - df["solar_ratio"]).clip(lower=0)
        df["is_low_solar"] = (df["solar_ratio"] < 0.70).fillna(0).astype("int8")
        df["is_re_adverse"] = ((df["is_rain_day"] == 1) | (df["is_low_solar"] == 1)).astype("int8")
        wk = df["is_weekend"] if "is_weekend" in df.columns else 0
        df["is_rain_weekday"] = ((df["is_rain_day"] == 1) & (wk.fillna(0) == 0)).astype("int8")
        # 운영가용 구분 — 일사 파생은 예보 확보 불가로 분석 전용
        for c, dfn, op, nt in (
            ("rain_daytime", "주간(9~17시) 강수합(mm)", "Y", WNOTE_RE),
            ("is_rain_day", "강수일(주간강수합≥1mm)", "Y", WNOTE_RE),
            ("is_rain_weekday", "평일 강수일", "Y", WNOTE_RE),
            ("solar_daytime", "주간(9~17시) 일사합(MJ/m²)", "N", WNOTE_SOLAR),
            ("solar_ratio", "주간일사 / 동월 직전60일 중앙값", "N", WNOTE_SOLAR),
            ("solar_deficit", "일사 부족분 max(0, 1-비율)", "N", WNOTE_SOLAR),
            ("is_low_solar", "저일사일(비율<0.70)", "N", WNOTE_SOLAR),
            ("is_re_adverse", "재생불리일(강수 또는 저일사)", "N", WNOTE_SOLAR),
        ):
            add_dict(c, "기상", dfn, op, nt)
        pv_p = os.path.join(BASE_DIR, "data", "interim", "pv_forecast_hourly.csv")
        if os.path.exists(pv_p):
            pv = pd.read_csv(pv_p, encoding="utf-8-sig")
            df = df.merge(pv[["date", "hour_n", "pv_fcst"]], on=["date", "hour_n"], how="left")
            add_dict("pv_fcst", "신재생", "육지 태양광 출력 전망(MWh)", "Y",
                     "자체 전망(단기예보 가용 기상) — 2026.9.1 운영 투입")
            if "demand_fcst_land" in df.columns:
                df["net_load_fcst"] = df["demand_fcst_land"] - df["pv_fcst"]
                # 2026.9.8 A-5 재판정 — 시간별 D+1(18·21·23:30시) 조건에서는 확보 가능
                #  · 종전 'N' 판정은 **주간예측(15시)** 기준 — 시간별 예측에 부적합
                #  · demand_fcst_land 는 이미 운영 투입(중요도 3위) → 정합성 확보
                #  · 주간예측은 --no-demand 로 demand_fcst*·net_load* 동시 제외
                _op = "Y" if args.netload == "on" else "N"
                _nt = ("2026.9.8 A-5 운영 투입 — 시간별 D+1 조건 확보"
                       if _op == "Y" else
                       "2026.9.1 분석 전용 — 수요예측 의존, 예측 시각 미확보")
                add_dict("net_load_fcst", "신재생",
                         "순부하 전망 = 육지 수요예측 − 태양광 전망(MW)", _op, _nt)

                # ── A-5 순부하 파생 6종 (2026.9.8 신설) ──────────────
                #  · 배경 : 2026.9.8 13시 순부하 일중 최저(54,111MW) 구간에서
                #           실적 SMP 107.58원 급락 / 예측 168.81원 — 오차 +61.23원
                #  · 취지 : 절대 수준이 아닌 **일중 상대 위치**를 신호로 제공
                #           (태양광 침투에 따른 Duck Curve 추종)
                if args.netload == "on":
                    g = df.groupby("date")["net_load_fcst"]
                    df["net_load_daily_mean"] = g.transform("mean")
                    df["net_load_daily_min"] = g.transform("min")
                    df["net_load_ratio"] = (df["net_load_fcst"]
                                            / df["net_load_daily_mean"])
                    df["net_load_rank"] = g.rank(method="min").astype("float32")
                    df["is_net_load_trough"] = (
                        df["net_load_rank"] <= 3).astype("int8")
                    df["pv_penetration"] = (df["pv_fcst"]
                                            / df["demand_fcst_land"])
                    df["net_load_diff_d1"] = (df["net_load_fcst"]
                                              - df["net_load_fcst"].shift(24))
                    for c, dfn in (
                        ("net_load_daily_min", "당일 순부하 전망 최저(MW)"),
                        ("net_load_ratio", "순부하 / 당일 평균 — 일중 상대수준"),
                        ("net_load_rank", "당일 24시간 중 순부하 오름차순 순위"),
                        ("is_net_load_trough", "순부하 최저 3개 시간대 여부"),
                        ("pv_penetration", "태양광 침투율 = 태양광전망 / 수요예측"),
                        ("net_load_diff_d1", "순부하 전일 동시간 대비 변화(MW)"),
                    ):
                        add_dict(c, "신재생", dfn, "Y",
                                 "2026.9.8 A-5 신설 — Duck Curve 추종")
                    add_dict("net_load_daily_mean", "신재생",
                             "당일 순부하 전망 평균(MW)", "N",
                             "2026.9.8 A-5 — 파생 산출용, 직접 투입 제외")
                    rep("순부하 변수군", "투입 7종",
                        f"최저 {df['net_load_fcst'].min():,.0f}MW / "
                        f"침투율 최대 {df['pv_penetration'].max() * 100:.1f}%")
            rep("태양광 전망", f"결합 {df['pv_fcst'].notna().mean() * 100:.1f}%",
                f"{int(df['pv_fcst'].notna().sum()):,}행")
        else:
            rep("태양광 전망", "미결합",
                "python scripts\\step8_pv_forecast.py --emit 선행")
        rep("재생불리 변수", "강수 3종 + pv_fcst 운영 / 일사 6종·순부하 분석",
            f"강수일 비중 {df['is_rain_day'].mean() * 100:.1f}%")

    # ── ④-3 연료비 구조 변수 ───────────────────
    #  · 근거 : 2026.8.27 사내 시연 지적 ①·② — 시간별·주간에 미래 연료가 미반영
    #  · 원천 : step3_fuel_features.py 산출(월별 변동비·급전순위 곡선·구조모델 SMP)
    #  · 가용 : 발전기 제원 보유 구간(2025-01~) 한정 → 이전 구간은 결측 유지
    #           (LightGBM 결측 분기 처리로 전 구간 표본 유지 — 실증설계 ①)
    # ── ④-2′ 급전순위 공개자료 연료비 변수 (2026.9.1 채택) ──────────
    #  · 원천 : 전력거래소 「변동비순위」 월별 공개자료(step7_parse_meritfile.py)
    #  · 근거 : 보유율 78.0%(종전 내부 제원 29.4%)로 확대되어 재실증한 결과
    #           양 홀드아웃 구간 모두 개선 — 26년 -0.17%p / 25년 -0.20%p
    #  · 운영 : 변동비순위는 적용 전월에 사전 공표되어 예측 시점 확보 가능
    if args.merit_feat == "on":
        print("\n■ ④-2′ 연료비 변수 — 급전순위 공개자료 결합")
        mp = os.path.join(BASE_DIR, "data", "interim", "merit_features_monthly.csv")
        if not os.path.exists(mp):
            rep("연료비 변수", "원천 부재 — 생성 생략",
                "python scripts\\step3_merit_features.py 선행 실행 필요")
        else:
            mf = pd.read_csv(mp, encoding="utf-8-sig")
            mcols = [c for c in mf.columns if c.startswith("pm_")]
            df["ym"] = df["date"].astype(str).str[:7]
            df = df.merge(mf[["ym"] + mcols], on="ym", how="left")
            DESC = {
                "pm_mc_q60": "급전순위 곡선 누적용량 60% 지점 변동비(원/kWh)",
                "pm_vc_mean": "중앙급전발전기 평균 변동비(원/kWh)",
                "pm_vc_lng": "LNG 발전기 평균 변동비(원/kWh)",
                "pm_mc_q60_chg": "한계변동비 전월 대비 변화율",
            }
            for c in mcols:
                add_dict(c, "연료비", DESC.get(c, c), "Y",
                         "전력거래소 공개자료 — 적용 전월 사전 공표")
            rep("연료비 변수", f"{len(mcols)}종 결합",
                f"보유율 {df[mcols[0]].notna().mean() * 100:.1f}% / "
                f"{mf['ym'].min()}~{mf['ym'].max()}")

    # ── ④-2″ 경험 가산분 (2026.10.4, KPX 05번 NLP·SUP 대리) ──────────
    if args.markup_feat == "on":
        print("\n■ ④-2″ 경험 가산분 변수 — 가격결정기 변동비 대비 시차")
        mp = os.path.join(BASE_DIR, "data", "interim", "markup_features_hourly.csv")
        if not os.path.exists(mp):
            try:
                from step3_markup_features import build_markup, OUT as MOUT
                feat_m = build_markup()
                os.makedirs(os.path.dirname(MOUT), exist_ok=True)
                feat_m.to_csv(MOUT, index=False, encoding="utf-8-sig")
                mp = MOUT
                rep("가산분 원천", "자동 생성", MOUT)
            except Exception as e:
                rep("가산분 변수", "생성 실패 — 생략", f"{type(e).__name__}: {e}")
                mp = None
        if mp and os.path.exists(mp):
            mk = pd.read_csv(mp, encoding="utf-8-sig")
            mcols = [c for c in mk.columns if c.startswith("ps_markup_")]
            df = df.merge(mk[["date", "hour_n"] + mcols],
                          on=["date", "hour_n"], how="left")
            DESC_M = {
                "ps_markup_lag_d1": "전일 동시간 경험가산분(SMP−가격결정기 변동비)",
                "ps_markup_ma7": "동일시간대 직전7일 가산분 평균",
                "ps_markup_ma14": "동일시간대 직전14일 가산분 평균",
                "ps_markup_dmean_lag": "전일 가산분 일평균",
            }
            for c in mcols:
                add_dict(c, "연료비", DESC_M.get(c, c), "Y",
                         "2026.10.4 — KPX 05번 무부하·기동가격 경험대리 / lag≥24h")
            rep("가산분 변수", f"{len(mcols)}종 결합",
                f"보유율 {df[mcols[0]].notna().mean() * 100:.1f}%")

    if args.fuel_feat == "on":
        print("\n■ ④-3 연료비 구조 변수 — 월별 산정 로직 결합")
        from step3_fuel_features import marginal_cost, OUT_MONTHLY, OUT_CURVE
        if not (os.path.exists(OUT_MONTHLY) and os.path.exists(OUT_CURVE)):
            rep("연료비 변수", "원천 부재 — 생성 생략",
                "python scripts\\step3_fuel_features.py 선행 실행 필요")
        else:
            fm = pd.read_csv(OUT_MONTHLY, encoding="utf-8-sig")
            fcols = [c for c in fm.columns if c.startswith("fc_")]
            df["ym"] = df["date"].astype(str).str[:7]
            df = df.merge(fm[["ym"] + fcols], on="ym", how="left")

            # ① 일유형별 구조모델 SMP — 평일·휴일 산출값 선택
            if {"fc_struct_weekday", "fc_struct_holiday"} <= set(df.columns):
                df["fc_struct_daytype"] = np.where(
                    df["daytype"].astype(str) == "평일",
                    df["fc_struct_weekday"], df["fc_struct_holiday"])
                add_dict("fc_struct_daytype", "연료비",
                         "일유형 대응 구조모델 월별 SMP(원/kWh)", "Y",
                         "급전순위 산출값 — 월 단위 사전 확정")

            # ② 시간대별 한계발전기 변동비 — 급전순위 곡선 × 부하 대리지표
            #    부하 대리 : 전일 동시간 수요예측(lag24) — 예측 시점 확보 가능
            curve = pd.read_csv(OUT_CURVE, encoding="utf-8-sig")
            load_proxy = df["demand_fcst_total"].shift(24)
            df["fc_mc_hour"] = marginal_cost(curve, df["ym"].to_numpy(),
                                             load_proxy.to_numpy())
            df["fc_mc_gap"] = df["fc_mc_hour"] - df["fc_vc_cap_wavg"]
            add_dict("fc_mc_hour", "연료비",
                     "시간대별 한계발전기 변동비(원/kWh)", "Y",
                     "부하 대리지표 = 전일 동시간 수요예측(lag24)")
            add_dict("fc_mc_gap", "연료비",
                     "한계발전기 변동비 - 용량가중 평균변동비", "Y",
                     "급전 여건 긴장도 대리지표")

            # ③ 월별 원천 변수 사전 등록
            FDEF = {
                "fc_unit_n": "집계 대상 발전기 수(기)",
                "fc_cap_total": "총 가용용량(MW)",
                "fc_vc_mean": "발전기 단순평균 변동비(원/kWh)",
                "fc_vc_cap_wavg": "용량가중 평균 변동비(원/kWh)",
                "fc_vc_p50": "용량가중 변동비 중위값(원/kWh)",
                "fc_vc_p75": "용량가중 변동비 75분위(원/kWh)",
                "fc_vc_p90": "용량가중 변동비 90분위(원/kWh)",
                "fc_vc_lng": "LNG 평균 변동비(원/kWh)",
                "fc_vc_coal": "석탄 평균 변동비(원/kWh)",
                "fc_vc_nuclear": "원자력 평균 변동비(원/kWh)",
                "fc_vc_oil": "유류 평균 변동비(원/kWh)",
                "fc_cap_ratio_lng": "LNG 용량 비중",
                "fc_cap_ratio_coal": "석탄 용량 비중",
                "fc_cap_ratio_nuclear": "원자력 용량 비중",
                "fc_cap_ratio_oil": "유류 용량 비중",
                "fc_vc_spread_lng_coal": "LNG-석탄 변동비 격차(원/kWh)",
                "fc_struct_weekday": "구조모델 평일 SMP(원/kWh)",
                "fc_struct_holiday": "구조모델 휴일 SMP(원/kWh)",
                "fc_struct_total": "구조모델 종합 SMP(원/kWh)",
            }
            for c in fcols:
                d0 = FDEF.get(c)
                if d0 is None:
                    if c.endswith("_chg"):
                        d0 = f"{FDEF.get(c[:-4], c[:-4])} 전월 대비 변화율"
                    elif c.endswith("_ma3"):
                        d0 = f"{FDEF.get(c[:-4], c[:-4])} 3개월 이동평균"
                    else:
                        d0 = c
                add_dict(c, "연료비", d0, "Y",
                         "월 단위 사전 확정 — 연료가 전망 기반")

            made = [c for c in df.columns if c.startswith("fc_")]
            cov = df["fc_vc_cap_wavg"].notna().mean() * 100
            rep("생성 변수", f"연료비 구조 {len(made)}종")
            rep("변수 보유율", f"{cov:.1f}%",
                f"발전기 제원 보유 구간 {fm['ym'].min()}~ 한정")
            rep("한계비용 산출률", f"{df['fc_mc_hour'].notna().mean() * 100:.1f}%",
                "급전순위 곡선 × 부하 대리지표")
            df = df.drop(columns=["ym"])

    # ── ⑤ 실적계 변수(분석 전용) ───────────────
    print("\n■ ⑤ 실적계 변수 — 분석 전용(operational=N)")
    df["net_load"] = df["demand_actual"] - df[["pv_total", "wind_total"]].sum(axis=1, min_count=1)
    df["re_ratio"] = (df[["pv_total", "wind_total"]].sum(axis=1, min_count=1)
                      / df["demand_actual"])
    for c, d in (("demand_actual", "전력수요 실적(MWh)"),
                 ("pv_total", "태양광 발전량 실적(MWh)"),
                 ("wind_total", "풍력 발전량 실적(MWh)"),
                 ("pv_land", "육지 태양광"), ("pv_jeju", "제주 태양광"),
                 ("wind_land", "육지 풍력"), ("wind_jeju", "제주 풍력"),
                 ("net_load", "순부하 = 수요실적 - 신재생"),
                 ("re_ratio", "신재생 침투율")):
        add_dict(c, "실적", d, "N", "공표 지연으로 운영 시점 미가용 — 분석·해석용")
    add_dict("pv_source", "메타", "태양광 출처(파일 확정 / API 잠정)", "-")
    for c in ("unit_id", "unit_name", "grp"):
        add_dict(c, "참고", "가격결정발전기 정보(사내자료)", "N", "보유 구간 한정【B】")
    rep("생성 변수", "실적 파생 2종", "net_load, re_ratio")

    # ── ⑥ 저장 ─────────────────────────────────
    print("\n■ ⑥ 저장")
    df = df.drop(columns=[c for c in ("holiday_name",) if c in df.columns])
    front = ["date", "hour_n", "ts", "smp_land", "smp_jeju"]
    cols = front + [c for c in df.columns if c not in front]
    out = df[cols]
    save_atomic(out, OUT)
    rep("피처셋", f"{len(out):,}행 × {len(out.columns)}열")
    print(f"  → 저장 {OUT}")

    dd = pd.DataFrame(DICT_ROWS).drop_duplicates(subset=["변수명"])
    # 미등록 변수 보완
    miss = [c for c in out.columns if c not in set(dd["변수명"])]
    for c in miss:
        dd.loc[len(dd)] = {"변수명": c, "구분": "기타", "정의": "", "운영가용": "-", "비고": ""}
    save_atomic(dd, DICT)
    rep("변수사전", f"{len(dd)}개 변수")
    print(f"  → 저장 {DICT}")

    n_op = int((dd["운영가용"] == "Y").sum())
    n_no = int((dd["운영가용"] == "N").sum())
    rep("운영 가용 변수", f"{n_op}개", "모델 입력 후보")
    rep("운영 불가 변수", f"{n_no}개", "분석 전용")

    lp = os.path.join(LOG_DIR, f"step3_features_report_{datetime.now():%Y%m%d_%H%M%S}.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(lp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["항목", "값", "비고"])
        w.writeheader()
        w.writerows(REPORT)
    print(f"  → 보고서 {lp}")
    print("\n" + "=" * 70)
    print(" ※ 학습 시 operational='Y' 변수만 사용. 'N' 변수는 오차분석·해석 전용")
    print("=" * 70)


if __name__ == "__main__":
    main()
