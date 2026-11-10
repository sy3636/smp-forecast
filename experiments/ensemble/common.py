# -*- coding: utf-8 -*-
"""
common.py — 앙상블 실험 공통 모듈 (경로·데이터·구간·지표)
==============================================================================
구간 (모든 스크립트 공통)
  · 표본 외 예측 : 2022-01-01 ~ (2020~2021년은 학습 전용)
  · 개발(dev)    : 2020-01-01 ~ 2024-12-31 — 경보 비율·주의/경보 기준은 여기서만 결정 (2026-11-08 팀 작업본에서 변경, 이전 2024~2025)
                   ※ 예측·확률이 실제로 있는 것은 2022년부터라 개발 성적은 사실상 2022~2024
  · 시험(test)   : 2025-01-01 ~ 2026-10-31 — 결정이 끝난 뒤 평가 (이전 2026-01-01~10-02)
  · 두 구간 모두 config/eval_settings.json 의 train·test 값 (2026-11-08부터 설정으로 바꿈)
  · 부서 홀드아웃 : 2026-01-29 ~ 2026-09-02 — 실행계획서 수치와 비교용

지표 (scripts/step8_rmae.py 와 동일)
  · MAE, rMAE = MAE ÷ MAE(naive-1 전일 동시간), MAPE·편의 분모 = max(실적, 20)

예측 시점 가정 (D+1 시간별)
  · D일 저녁(21시) — D일 SMP 24시간·D+1 수요예측 확보, D+1 SMP 미확보
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
OUT = BASE / "outputs" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

FLOOR = 20.0
OOS_START = "2022-01-01"
# 학습(개발)·시험 구간은 config/eval_settings.json 에서 읽음 (양끝 포함 → 여기서는 [시작, 끝+1일) 로 바꿔 씀)
def _settings():
    import json
    from datetime import date, timedelta
    s = json.loads((BASE / "config" / "eval_settings.json").read_text(encoding="utf-8"))
    nxt = lambda d: (date.fromisoformat(d) + timedelta(days=1)).isoformat()
    return s, (s["train"][0], nxt(s["train"][1])), (s["test"][0], nxt(s["test"][1]))


SETTINGS, DEV, TEST = _settings()
HOLD = ("2026-01-29", "2026-09-03")
FRESH = ("2026-10-03", TEST[1])        # 새 시험 (끝 = 시험 끝, 2026-10-31) — collect_recent.py 로 받은, 개선 전 어떤 결정에도 쓰지 않은 기간
MIDDAY = [11, 12, 13, 14, 15]


def load_lightgbm():
    """torch 에 들어 있는 libomp 를 먼저 올린 뒤 lightgbm 을 불러옴 (Homebrew libomp 없는 맥 대비)"""
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except ImportError:
        pass
    import lightgbm
    return lightgbm


def rd(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kw)


CAL_COLS = ["is_prev_day_off", "is_next_day_off", "holiday_run_len", "holiday_position",
            "is_bridge_day", "is_major_holiday", "is_major_holiday_period", "is_major_holiday_eve",
            "is_kpx_myeong_adj", "is_kpx_holiday_eve", "is_kpx_holiday_return"]
RECENT_SMP = DATA / "interim" / "api_smp_recent.csv"      # collect_recent.py — 데이터셋 이후 SMP·수요예측
RECENT_TEMP = DATA / "interim" / "asos_temp_recent.csv"   # collect_recent.py — ASOS 기온 실측


def calendar_features(days, off, hol_name):
    """연휴 구조 변수 (smp_features_smp_land.csv 와 같은 정의 — 데이터셋 이후 날짜를 채우는 데 사용)"""
    n = len(days)
    off = np.asarray(off, int)
    major = np.array([any(k in str(s) for k in ("설날", "추석")) for s in hol_name], int)
    run_id = np.cumsum(np.r_[1, np.diff(off) != 0])
    out = {c: np.zeros(n, int) for c in CAL_COLS}
    out["is_prev_day_off"][1:] = off[:-1]
    out["is_next_day_off"][:-1] = off[1:]
    for r in np.unique(run_id[off == 1]):
        idx = np.where(run_id == r)[0]
        L = len(idx)
        out["holiday_run_len"][idx] = L
        out["holiday_position"][idx] = 1 if L == 1 else 3
        if L > 1:
            out["holiday_position"][idx[0]], out["holiday_position"][idx[-1]] = 2, 4
        if major[idx].any():
            out["is_major_holiday_period"][idx] = 1
            if idx[0] > 0:
                out["is_major_holiday_eve"][idx[0] - 1] = 1
            for k in (1, 2):                                    # 명절 연휴 전후 근무일 ±2
                for j in (idx[0] - k, idx[-1] + k):
                    if 0 <= j < n and off[j] == 0:
                        out["is_kpx_myeong_adj"][j] = 1
        if L >= 3:
            if idx[0] > 0 and off[idx[0] - 1] == 0:
                out["is_kpx_holiday_eve"][idx[0] - 1] = 1
            if idx[-1] + 1 < n and off[idx[-1] + 1] == 0:
                out["is_kpx_holiday_return"][idx[-1] + 1] = 1
    out["is_major_holiday"] = major
    out["is_bridge_day"] = ((off == 0) & (out["is_prev_day_off"] == 1) & (out["is_next_day_off"] == 1)).astype(int)
    return pd.DataFrame(out, index=days)


def load_hourly():
    """시간별 원천 + 달력 변수. 날짜 × 24시간 격자를 빠짐없이 만든다.
    collect_recent.py 로 받은 데이터셋 이후 자료(SMP·수요예측·기온)가 있으면 이어 붙인다."""
    d = rd(DATA / "processed" / "smp_dataset_hourly.csv")[
        ["date", "hour_n", "is_holiday", "holiday_name", "smp_land", "demand_fcst_land",
         "pv_land", "temp_land"]]
    d = d.rename(columns={"hour_n": "hour", "smp_land": "smp", "demand_fcst_land": "dem", "pv_land": "pv"})
    cal = rd(DATA / "processed" / "smp_features_smp_land.csv", usecols=["date", "hour_n"] + CAL_COLS)
    d = d.merge(cal.rename(columns={"hour_n": "hour"}), on=["date", "hour"], how="left")

    last = d["date"].max()
    rs = rd(RECENT_SMP).rename(columns={"hour_n": "hour"}) if RECENT_SMP.exists() else None
    if rs is not None:
        last = max(last, rs["date"].max())
    last = min(last, SETTINGS["data_end"])        # 모든 결과의 끝 = 설정의 data_end (2026-10-31) — 그 뒤 자료는 읽지 않음
    days = pd.date_range(d["date"].min(), last, freq="D").strftime("%Y-%m-%d")
    grid = pd.MultiIndex.from_product([days, range(1, 25)], names=["date", "hour"]).to_frame(index=False)
    d = grid.merge(d, on=["date", "hour"], how="left")
    for path, cols in [(RECENT_SMP, {"smp_land": "smp", "demand_fcst_land": "dem"}), (RECENT_TEMP, {"temp_land": "temp_land"})]:
        if path.exists():
            r = rd(path).rename(columns={"hour_n": "hour", **cols})[["date", "hour"] + list(cols.values())]
            d = d.merge(r, on=["date", "hour"], how="left", suffixes=("", "_new"))
            for c in cols.values():
                d[c] = d[c + "_new"].combine_first(d[c])      # API 값 우선
                d = d.drop(columns=c + "_new")

    hol = rd(DATA / "interim" / "api_holiday.csv")
    hol = hol[hol["is_holiday"] == 1]
    extra = d["date"].isin(hol["date"])
    d.loc[extra, "is_holiday"] = 1
    names = hol.drop_duplicates("date").set_index("date")["date_name"]
    d["holiday_name"] = d["holiday_name"].fillna(d["date"].map(names))
    d["is_holiday"] = d["is_holiday"].fillna(0).astype(int)
    dt = pd.to_datetime(d["date"])
    d["dow"] = dt.dt.dayofweek
    d["month"] = dt.dt.month
    d["off"] = ((d["dow"] >= 5) | (d["is_holiday"] == 1)).astype(int)
    d["ts"] = dt + pd.to_timedelta(d["hour"] - 1, unit="h")
    d = d.sort_values(["date", "hour"]).reset_index(drop=True)

    daily = d.iloc[::24]
    calc = calendar_features(daily["date"].to_numpy(), daily["off"].to_numpy(), daily["holiday_name"].to_numpy())
    for c in CAL_COLS:                                        # 데이터셋에 없는 날짜만 계산값으로 채움
        d[c] = d[c].combine_first(d["date"].map(calc[c]).astype(float))
    return d


def to_matrix(h, col):
    """시간별 열 → (일수, 24) 행렬"""
    return h[col].to_numpy(float).reshape(-1, 24)


def load_monthly():
    """월 단위 연료비 정보 — 변동비순위표(적용 전월 사전 공표), 국제 연료지수(2개월 지연 사용)"""
    m = rd(DATA / "interim" / "merit_features_monthly.csv")[["ym", "pm_mc_q60", "pm_vc_mean", "pm_vc_lng"]]
    f = rd(DATA / "interim" / "fuel_index_monthly.csv")[["ym", "idx_lng_krw", "idx_coal_krw", "dubai_usd_bbl"]]
    yms = pd.period_range("2020-01", "2026-12", freq="M").strftime("%Y-%m")
    out = pd.DataFrame({"ym": yms}).merge(m, on="ym", how="left")
    f = pd.DataFrame({"ym": yms}).merge(f, on="ym", how="left").set_index("ym").ffill()
    f = f.shift(2)                                   # 월평균 지수는 그 달이 끝나야 확정 → 2개월 지연
    return out.merge(f.reset_index(), on="ym", how="left").set_index("ym")


# ------------------------------------------------------------------ 지표
def metrics4(y, p, n1):
    y, p, n1 = (np.asarray(v, float) for v in (y, p, n1))
    m = ~(np.isnan(y) | np.isnan(p) | np.isnan(n1))
    y, p, n1 = y[m], p[m], n1[m]
    den = np.maximum(y, FLOOR)
    mae, mae1 = np.abs(p - y).mean(), np.abs(n1 - y).mean()
    return {"N": int(m.sum()), "MAE": mae, "rMAE": mae / mae1, "MAPE": (np.abs(p - y) / den).mean() * 100,
            "bias": ((p - y) / den).mean() * 100, "naive_MAE": mae1}


def interval_metrics(y, lo, hi):
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    m = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    y, lo, hi = y[m], lo[m], hi[m]
    return {"cover": ((y >= lo) & (y <= hi)).mean() * 100, "width": (hi - lo).mean(),
            "below": (y < lo).mean() * 100, "above": (y > hi).mean() * 100}


def period_mask(dates, period):
    dates = np.asarray(dates)
    return (dates >= period[0]) & (dates < period[1])


def fmt_metrics(r):
    return (f"N={r['N']:>6,}  MAE {r['MAE']:6.2f}  rMAE {r['rMAE']:.3f}  "
            f"MAPE {r['MAPE']:5.2f}%  편의 {r['bias']:+5.2f}%")
