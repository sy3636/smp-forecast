# -*- coding: utf-8 -*-
"""
step6_wx_pattern.py — 재생불리(강수·저일사) 시 9.1 패턴 반영
==============================================================================
목적
  · 2026.9.1 오차(심야 석탄 → 주간 LNG 점프)를 이후 예측에 재사용
  · LightGBM 점예측이 시차·수요에 평활될 때, 이례 조건에서만 형상을 당김

적용 조건 (모두 충족)
  · 평일
  · 주간 강수합 ≥ 1 mm 또는 일사비율 < 0.70
  · 거래시간 9~21시
  · 전일 동시간 SMP가 140원 미만(점프가 시차에 아직 없는 경우)이면 당김을 강화
    전일 동시간이 이미 고가면 시차가 패턴을 기억하므로 당김을 줄임

v1.1 (2026-09-14)
  · **고태양광 가드** : pv_penetration(또는 pv_fcst/수요)이 높으면
    일사비율 저일사 판정을 무시 — 9.14처럼 고PV인데 solar_ratio 오탐으로
    고가 형상(9.1)이 켜지는 오적용 차단
  · 강수 이례는 기존과 동일하게 유지

버전 : v1.1 (2026-09-14)
"""
import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = os.path.join(BASE_DIR, "config", "wx_re_pattern.json")

# 이 침투율 이상이면 해당 시각의 '저일사→고가' 신호를 신뢰하지 않음
PV_PEN_BLOCK_SOLAR = 0.045
# 주간(10~16) 최대 침투율이 이 이상이면 **당일 전체** 저일사 고가 판정 차단
PV_PEN_DAY_BLOCK = 0.060


def load_pattern():
    with open(PATTERN, encoding="utf-8") as f:
        return json.load(f)


def _penetration(pv_pen, pv_fcst, demand_fcst):
    if pv_pen is not None and pd.notna(pv_pen):
        try:
            return float(pv_pen)
        except (TypeError, ValueError):
            pass
    if pv_fcst is None or demand_fcst is None:
        return np.nan
    try:
        p, d = float(pv_fcst), float(demand_fcst)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(p) or not np.isfinite(d) or d <= 0:
        return np.nan
    return p / d


def _day_high_pv(pens, hours):
    """주간 고PV일이면 True — solar_ratio 오탐으로 9.1 고가 점프 차단."""
    vals = []
    for pen, h in zip(pens, hours):
        if pen is None or (isinstance(pen, float) and np.isnan(pen)):
            continue
        if 10 <= int(h) <= 16:
            vals.append(float(pen))
    if not vals:
        return False
    return max(vals) >= PV_PEN_DAY_BLOCK


def event_weight(rain_daytime, solar_ratio, is_weekend, hour_n, lag_d1, cfg,
                 pv_pen=None, pv_fcst=None, demand_fcst=None, day_high_pv=False):
    """0~alpha_max. 이례가 아니면 0."""
    if int(is_weekend or 0) == 1:
        return 0.0
    h = int(hour_n)
    if h < int(cfg["hour_lo"]) or h > int(cfg["hour_hi"]):
        return 0.0
    rain = float(rain_daytime) if pd.notna(rain_daytime) else 0.0
    sol = float(solar_ratio) if pd.notna(solar_ratio) else 1.0
    pen = _penetration(pv_pen, pv_fcst, demand_fcst)
    # 고PV이면 일사비율 기반 저일사 판정 차단 (강수만 유효)
    sol_adverse = sol < float(cfg["solar_ratio_cut"])
    if sol_adverse and day_high_pv:
        sol_adverse = False
    elif sol_adverse and pd.notna(pen) and pen >= PV_PEN_BLOCK_SOLAR:
        sol_adverse = False
    rain_adverse = rain >= float(cfg["rain_mm"])
    if not (rain_adverse or sol_adverse):
        return 0.0
    # 저일사일수록 강하게. 9.1 일사비율 미확정이므로 강수량도 병행
    w_sol = (float(np.clip((float(cfg["solar_ratio_cut"]) - sol) / 0.50, 0.0, 1.0))
             if sol_adverse else 0.0)
    w_rain = float(np.clip(rain / 8.0, 0.0, 1.0)) if rain_adverse else 0.0
    w = max(w_sol, w_rain, 0.45)   # 이례로 판정되면 최소 45%
    lag = float(lag_d1) if pd.notna(lag_d1) else 0.0
    if lag >= float(cfg["lag_jump_cut"]):
        w *= 0.35                  # 시차가 이미 고가 — 이중 반영 억제
    return float(np.clip(w * float(cfg["alpha_max"]), 0.0, float(cfg["alpha_max"])))


def apply_pattern(pred, hour_n, rain_daytime, solar_ratio, is_weekend,
                  lag_d1=None, pv_pen=None, pv_fcst=None, demand_fcst=None):
    """예측값 배열을 9.1 실적 형상과 혼합. 길이 동일한 1차원."""
    cfg = load_pattern()
    shape = {int(k): float(v) for k, v in cfg["actual_by_hour"].items()}
    pred = np.asarray(pred, dtype=float).copy()
    hour_n = np.asarray(hour_n)
    n = len(pred)
    rain = np.asarray(rain_daytime) if np.ndim(rain_daytime) else np.repeat(rain_daytime, n)
    sol = np.asarray(solar_ratio) if np.ndim(solar_ratio) else np.repeat(solar_ratio, n)
    wk = np.asarray(is_weekend) if np.ndim(is_weekend) else np.repeat(is_weekend, n)
    if lag_d1 is None:
        lag = np.full(n, np.nan)
    else:
        lag = np.asarray(lag_d1) if np.ndim(lag_d1) else np.repeat(lag_d1, n)

    def _arr(x):
        if x is None:
            return [None] * n
        a = np.asarray(x)
        if a.ndim == 0:
            return [a.item()] * n
        return list(a)

    pens = _arr(pv_pen)
    pvs = _arr(pv_fcst)
    dems = _arr(demand_fcst)
    # 침투율 벡터 (일 단위 고PV 판정용)
    pen_vals = []
    for i in range(n):
        pen_vals.append(_penetration(pens[i], pvs[i], dems[i]))
    day_hi = _day_high_pv(pen_vals, hour_n)

    for i in range(n):
        w = event_weight(rain[i], sol[i], wk[i], hour_n[i], lag[i], cfg,
                         pv_pen=pens[i], pv_fcst=pvs[i], demand_fcst=dems[i],
                         day_high_pv=day_hi)
        if w <= 0:
            continue
        tgt = shape.get(int(hour_n[i]))
        if tgt is None:
            continue
        pred[i] = (1.0 - w) * pred[i] + w * tgt
    return np.round(pred, 2)
