# -*- coding: utf-8 -*-
"""
kpx_daytype.py — KPX 준용 일유형 분류 (공유 모듈)
==============================================================================
출처 : 중급 전력거래반 교재 07번 54·55쪽 일유형 매핑
용도 : step3_features(피처) · step6_bias_adjust(편의창 제외) · step8_kpx_reference_check

버전 : v1.0 (2026-10-04)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 편의보정 「평일」 창에서 제외할 KPX 특수 근무일 (비교군 K1)
EXCLUDE_FROM_WEEKDAY = ("명절 전후", "징검다리 평일")


def build_calendar(d: pd.DataFrame) -> pd.DataFrame:
    """일자별 휴무·공휴일명 → KPX 준용 일유형 및 현행 3분류.

    필수 열 : date, daytype
    선택 열 : holiday_name (없으면 공휴일명 기반 세분 불가 → 주말·평일만)
    """
    cols = ["date", "daytype"]
    if "holiday_name" in d.columns:
        cols.append("holiday_name")
    cal = d[cols].drop_duplicates("date").copy().reset_index(drop=True)
    if "holiday_name" not in cal.columns:
        cal["holiday_name"] = np.nan
    cal["dt"] = pd.to_datetime(cal["date"])
    cal["dow"] = cal["dt"].dt.weekday
    cal["off"] = (cal["daytype"].astype(str) == "휴일") | (cal["dow"] >= 5)
    cal["pub"] = (cal["holiday_name"].notna()
                  & (cal["holiday_name"].astype(str).str.len() > 0))
    cal["myeong"] = (cal["holiday_name"].fillna("")
                     .astype(str).str.contains("설날|추석"))
    off = cal["off"]
    my = cal["myeong"]
    pub = cal["pub"]
    dates = list(cal["date"])
    n = len(dates)

    def run_len(i, step):
        cnt, j = 0, i + step
        while 0 <= j < n and bool(off.iloc[j]):
            cnt += 1
            j += step
        return cnt

    def near_myeong(i):
        return any(0 <= i + k < n and bool(my.iloc[i + k])
                   for k in (-2, -1, 1, 2))

    cls = []
    for i, r in cal.iterrows():
        if bool(r["myeong"]):
            c = "명절 연휴"
        elif bool(r["pub"]):
            c = {0: "공휴일(월)", 5: "공휴일(토)", 6: "일요일"}.get(
                int(r["dow"]), "공휴일(화–금)")
        elif int(r["dow"]) == 6:
            c = "일요일"
        elif int(r["dow"]) == 5:
            long_wk = (run_len(i, -1) + run_len(i, 1) + 1) >= 3
            near_pub = ((i > 0 and bool(pub.iloc[i - 1]))
                        or (i + 2 < n and bool(pub.iloc[i + 2])))
            c = "토요일(연휴 중)" if long_wk and near_pub else "토요일"
        else:
            prev_run, next_run = run_len(i, -1), run_len(i, 1)
            if near_myeong(i):
                c = "명절 전후"
            elif prev_run >= 1 and next_run >= 1:
                c = "징검다리 평일"
            elif prev_run >= 3:
                c = "연휴 익일"
            elif int(r["dow"]) == 0:
                c = "월요일"
            elif next_run >= 3:
                c = "연휴 전일"
            else:
                c = "화–금"
        cls.append(c)
    cal["kpx_class"] = cls
    cal["cls3"] = np.where(
        cal["pub"] | (cal["dow"] == 6), "일공휴",
        np.where(cal["dow"] == 5, "토", "평일"))
    return cal[["date", "dow", "off", "pub", "kpx_class", "cls3"]]


def attach_kpx_flags(df: pd.DataFrame) -> pd.DataFrame:
    """시간별 프레임에 KPX 일유형 이진 플래그 결합."""
    cal = build_calendar(df)
    out = df.merge(cal[["date", "kpx_class"]], on="date", how="left")
    out["is_kpx_myeong_adj"] = (out["kpx_class"] == "명절 전후").astype("int8")
    out["is_kpx_bridge"] = (out["kpx_class"] == "징검다리 평일").astype("int8")
    out["is_kpx_holiday_eve"] = (out["kpx_class"] == "연휴 전일").astype("int8")
    out["is_kpx_holiday_return"] = (out["kpx_class"] == "연휴 익일").astype("int8")
    out["is_kpx_special_weekday"] = (
        out["kpx_class"].isin(EXCLUDE_FROM_WEEKDAY)).astype("int8")
    return out
