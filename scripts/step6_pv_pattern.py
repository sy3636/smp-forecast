# -*- coding: utf-8 -*-
"""
step6_pv_pattern.py — 고태양광·경부하 → SMP 하향 연화 (v1.1)
==============================================================================
배경
  · 2026.9.14 : 주간 PV 전망 높음에도 운영·혼성이 실적 대비 크게 과대
  · 주간(9~17) MAE ≫ 야간, PV전망↔과대오차 상관 +0.68
  · 기존 9.1 패턴은 **저일사·강수 → 고가** 방향 — 고PV 국면과 반대
  · 2026.10.4 : KPX 05번 특수경부하 0원 + 관찰 9.25~27 낮 0원
                → light_load_regime 시 연화 강화·하한 목표 적용

설계
  · 운영 가용 신호: pv_penetration = pv_fcst / demand_fcst_land (피처 Y)
  · 주간·고침투율일수록 예측을 전일 시차(+여유) 쪽으로 **하향만** 연화
  · 경부하 레짐이면 alpha 가산·lag_margin 축소·floor_target 허용
    — light_load.enabled=false 시 해제(경부하 시간도 기본 연화 규칙만 적용)
  · 상향은 하지 않음(과소 구간은 편의보정·모델에 위임)

설정 : config/pv_soft_pattern.json

연동 : step6_predict / step6_weekly_predict — 9.1패턴 직후·편의보정 직전

버전 : v1.2 (2026-10-05) — 경부하 강화 on/off(light_load.enabled), 운영 해제
         10.4·10.5 연휴 실측 한낮 89~107원 — 레짐 표시 시간 과소예측(10.5 모의 적용 시
         11~16시 |잔차| 76.5원). 레짐 판정 검증 전까지 해제
       v1.1 (2026-10-04) — 경부하 0원 레짐 연화
       v1.0 (2026-09-14)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = os.path.join(BASE_DIR, "config", "pv_soft_pattern.json")


def load_pattern():
    with open(PATTERN, encoding="utf-8") as f:
        return json.load(f)


def _pen_from_row(pv_fcst, demand_fcst):
    if pv_fcst is None or demand_fcst is None:
        return np.nan
    try:
        p, d = float(pv_fcst), float(demand_fcst)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(p) or not np.isfinite(d) or d <= 0:
        return np.nan
    return p / d


def soft_weight(pv_pen, hour_n, cfg, light_load=False):
    """0~alpha_max(·boost). 고PV 주간이 아니면 0. 경부하는 소량 PV에도 가산."""
    h = int(hour_n)
    if h < int(cfg["hour_lo"]) or h > int(cfg["hour_hi"]):
        return 0.0
    lo = float(cfg["pv_pen_lo"])
    hi = float(cfg["pv_pen_hi"])
    if hi <= lo:
        return 0.0
    pen = 0.0 if (pv_pen is None or (isinstance(pv_pen, float) and np.isnan(pv_pen))) else float(pv_pen)
    w = float(np.clip((pen - lo) / (hi - lo), 0.0, 1.0))
    a = float(cfg["alpha_max"])
    if light_load:
        ll = cfg.get("light_load") or {}
        # 경부하: 침투율 낮아도 최소 가중 부여
        w = max(w, 0.55)
        a = min(float(ll.get("alpha_cap", 0.85)),
                a + float(ll.get("alpha_boost", 0.35)))
    return float(w * a)


def apply_pv_softening(pred, hour_n, pv_pen=None, pv_fcst=None,
                       demand_fcst=None, lag_d1=None, light_load=None):
    """고PV·경부하 구간 예측 하향 연화.

    Returns (pred_out, note)
    """
    cfg = load_pattern()
    pred = np.asarray(pred, dtype=float).copy()
    hour_n = np.asarray(hour_n)
    n = len(pred)

    if pv_pen is None:
        if pv_fcst is None or demand_fcst is None:
            return np.round(pred, 2), "태양광연화 생략(신호없음)"
        pv_fcst = np.asarray(pv_fcst) if np.ndim(pv_fcst) else np.repeat(pv_fcst, n)
        demand_fcst = (np.asarray(demand_fcst) if np.ndim(demand_fcst)
                       else np.repeat(demand_fcst, n))
        pen = np.array([_pen_from_row(pv_fcst[i], demand_fcst[i]) for i in range(n)],
                       dtype=float)
    else:
        pen = np.asarray(pv_pen, dtype=float)
        if pen.ndim == 0:
            pen = np.repeat(float(pen), n)

    if lag_d1 is None:
        lag = np.full(n, np.nan)
    else:
        lag = np.asarray(lag_d1, dtype=float)
        if lag.ndim == 0:
            lag = np.repeat(float(lag), n)

    ll_cfg = cfg.get("light_load") or {}
    if light_load is None or not ll_cfg.get("enabled", True):
        ll_arr = np.zeros(n, dtype=bool)          # 경부하 강화 해제(v1.2) — 기본 연화만
    else:
        ll_arr = np.asarray(light_load)
        if ll_arr.ndim == 0:
            ll_arr = np.repeat(bool(ll_arr), n)
        else:
            ll_arr = ll_arr.astype(bool)

    margin_base = float(cfg.get("lag_margin", 8.0))
    min_drop_base = float(cfg.get("min_drop", 3.0))
    floor_tgt = float(ll_cfg.get("floor_target", 15.0))
    touched = 0
    drops = []
    n_ll = 0
    for i in range(n):
        is_ll = bool(ll_arr[i])
        w = soft_weight(pen[i], hour_n[i], cfg, light_load=is_ll)
        if w <= 0:
            continue
        margin = float(ll_cfg.get("lag_margin", 2.0)) if is_ll else margin_base
        min_drop = float(ll_cfg.get("min_drop", 1.0)) if is_ll else min_drop_base
        if np.isnan(lag[i]):
            tgt = pred[i] * (1.0 - 0.12 * w)
        else:
            tgt = float(lag[i]) + margin
        if is_ll:
            tgt = min(tgt, floor_tgt)
            n_ll += 1
        if pred[i] <= tgt + min_drop:
            continue
        new = (1.0 - w) * pred[i] + w * tgt
        drops.append(pred[i] - new)
        pred[i] = new
        touched += 1

    if not touched:
        return np.round(pred, 2), "태양광연화 없음"
    note = (f"태양광연화 {touched}h"
            f"(평균−{float(np.mean(drops)):.1f}원, maxPV침투 "
            f"{np.nanmax(pen) * 100:.1f}%")
    if n_ll:
        note += f", 경부하{n_ll}h"
    note += ")"
    return np.round(pred, 2), note
