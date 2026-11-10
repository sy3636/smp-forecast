# -*- coding: utf-8 -*-
"""
step8_chronos_shadow.py — Chronos-2 보조(shadow) 산출 (v1.0)
==============================================================================
역할
  · 현행 LightGBM(+편의보정) 게시는 **변경하지 않음**
  · Chronos-2(cov 우선, 실패 시 uni) 예측을 별도 이력에 기록 — 감시·앙상블 후보
  · torch/chronos 미설치·추론 실패 시 soft-fail (운영 중단 없음)

환경변수
  · CHRONOS_SHADOW=1     — 보조 산출 활성화(기본 1). 0이면 즉시 생략
  · CHRONOS_BLEND_W=0.0  — 게시 혼성 가중(0=현행 단독, 예: 0.3 → 70%현행+30%Chronos)
  · CHRONOS_BLEND_START  — 혼성 시험 시작일 YYYY-MM-DD (미지정=즉시)
  · CHRONOS_BLEND_END    — 혼성 시험 종료일 YYYY-MM-DD 포함 (미지정=무기한)
  · CHRONOS_MODEL        — HF id (기본 amazon/chronos-2)
  · CHRONOS_CONTEXT      — context 길이(기본 2048)
  · CHRONOS_DEVICE       — cpu|cuda|auto (기본 cpu)

시험 구간(권장, 2026.9.12 확정)
  · 2026-09-14 ~ 2026-09-27 / w=0.30
  · 근거: 8~9월 중첩 30일 스윕에서 w↑수록 MAE↓, 전면교체(w=1) 대신
    현행 주도(70%) 유지. 조기중단: 혼성 MAE가 현행 대비 +10% 이상 3일 연속

버전 : v1.1 (2026-09-12) — 혼성 시험 구간 게이트"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

_CFG_CACHE = None


def _load_cfg_file():
    """cron 은 compose 환경변수를 상속하지 않음 → config 파일 fallback."""
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    _CFG_CACHE = {}
    for name in ("chronos_trial.env", "chronos_trial.env.example"):
        path = os.path.join(CONFIG_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                _CFG_CACHE[k.strip()] = v.strip()
        break
    return _CFG_CACHE


def _cfg(name, default=None):
    if name in os.environ and os.environ[name] != "":
        return os.environ[name]
    return _load_cfg_file().get(name, default)


def _env_float(name, default):
    try:
        return float(_cfg(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(_cfg(name, default))
    except (TypeError, ValueError):
        return int(default)


def shadow_enabled():
    v = str(_cfg("CHRONOS_SHADOW", "1")).strip()
    return v not in ("0", "false", "off")


def blend_weight(day=None):
    """혼성 가중 — START 이전이거나 END(지정 시) 이후이면 0.
    END 미지정은 무기한(2026.9.27 운영 채택).

    day: 예측 대상일 YYYY-MM-DD (없으면 오늘 기준 구간만 검사)
    """
    w = _env_float("CHRONOS_BLEND_W", 0.0)
    w = max(0.0, min(1.0, w))
    if w <= 0:
        return 0.0
    start = (str(_cfg("CHRONOS_BLEND_START", "") or "")).strip()[:10]
    end = (str(_cfg("CHRONOS_BLEND_END", "") or "")).strip()[:10]
    ref = (str(day)[:10] if day
           else datetime.now().strftime("%Y-%m-%d"))
    if start and ref < start:
        return 0.0
    if end and ref > end:
        return 0.0
    return w


def available():
    """chronos 패키지 사용 가능 여부."""
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


def load_shadow_map(day, targets=None):
    """기 산출 shadow 파일에서 대상일 맵 구성 (torch 불필요)."""
    targets = targets or ["smp_land"]
    out = {}
    for t in targets:
        g = load_shadow_day(day, t)
        if g is not None and not g.empty and "예측" in g.columns:
            out[t] = g
    return out


def _save_hist(target, rec):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"chronos_shadow_{target}.csv")
    if os.path.exists(path):
        old = pd.read_csv(path, encoding="utf-8-sig")
        day = rec["date"].iloc[0]
        old = old[old["date"].astype(str).str[:10] != day]
        rec = pd.concat([old, rec], ignore_index=True)
    rec.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def produce(day, targets=None, model_id=None, context=None, device=None):
    """대상일 Chronos 예측 산출 → {target: DataFrame[date,hour_n,예측,실적,구성]}

    실패 시 예외 — 호출측에서 soft-fail.
    """
    if not shadow_enabled():
        raise RuntimeError("CHRONOS_SHADOW=0")
    if not available():
        raise ImportError("chronos 미설치 — pip install -r requirements-chronos.txt")

    from step8_chronos_test import (  # noqa: E402
        load_series, build_pipeline, forecast_day,
    )

    model_id = model_id or _cfg("CHRONOS_MODEL", "amazon/chronos-2")
    context = context or _env_int("CHRONOS_CONTEXT", 2048)
    device = device or _cfg("CHRONOS_DEVICE", "cpu")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    targets = targets or ["smp_land"]
    pipe = build_pipeline(model_id, device)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_map = {}

    for t in targets:
        hist, cov_col = load_series(t)
        # D+1 등 실적 미발생일 — 24시간 골격 추가(실적 NaN, 수요예측은 피처에서)
        if hist[hist["date"] == day].empty:
            base = pd.to_datetime(day)
            fut = pd.DataFrame({
                "date": day,
                "hour_n": list(range(1, 25)),
                t: np.nan,
            })
            fut["timestamp"] = base + pd.to_timedelta(fut["hour_n"] - 1, unit="h")
            fut["naive"] = np.nan
            # 공변량(수요예측) — 피처셋에서 확보 시 cov 가능
            if cov_col:
                try:
                    feat = pd.read_csv(
                        os.path.join(BASE_DIR, "data", "processed",
                                     f"smp_features_{t}.csv"),
                        encoding="utf-8-sig",
                        usecols=["date", "hour_n", cov_col])
                    feat["date"] = feat["date"].astype(str).str[:10]
                    g = feat[feat["date"] == day][["hour_n", cov_col]]
                    fut = fut.merge(g, on="hour_n", how="left")
                except Exception:
                    fut[cov_col] = np.nan
            hist = pd.concat([hist, fut], ignore_index=True)
        mode = "cov"
        df, why = forecast_day(pipe, hist, day, t, cov_col, mode, context)
        if df is None and mode == "cov":
            mode = "uni"
            df, why = forecast_day(pipe, hist, day, t, cov_col, mode, context)
        if df is None:
            raise RuntimeError(f"{t} 산출 실패 — {why}")
        rec = df[["date", "hour_n", "예측", "실적"]].copy()
        rec.insert(0, "예측생성시각", ts)
        rec["구성"] = mode
        rec["모델"] = model_id
        rec["예측"] = np.round(rec["예측"].astype(float), 2)
        _save_hist(t, rec)
        out_map[t] = rec
    return out_map


def apply_blend(results, chronos_map, weight=None, day=None):
    """results[t]['예측'] 에 Chronos 혼성 적용. weight=0이면 무변경.

    returns (applied:bool, note:str)
    """
    w = (blend_weight(day) if weight is None
         else max(0.0, min(1.0, float(weight))))
    if w <= 0 or not chronos_map:
        why = "shadow 전용(게시=현행)"
        if weight is None and _env_float("CHRONOS_BLEND_W", 0.0) > 0:
            why = f"시험구간 외(게시=현행, 설정w={_env_float('CHRONOS_BLEND_W', 0):.2f})"
        return False, why
    notes = []
    for t, out in results.items():
        if t not in chronos_map:
            continue
        c = chronos_map[t][["hour_n", "예측"]].rename(columns={"예측": "_c"})
        m = out.merge(c, on="hour_n", how="left")
        if m["_c"].isna().any():
            notes.append(f"{t}:결측스킵")
            continue
        out["예측"] = np.round((1 - w) * m["예측"].to_numpy(float)
                               + w * m["_c"].to_numpy(float), 2)
        notes.append(f"{t}:w={w}")
    return True, " / ".join(notes) if notes else "대상 없음"


def load_shadow_day(day, target="smp_land"):
    """대시보드용 — 특정일 shadow 24시간 (없으면 None)."""
    path = os.path.join(OUT_DIR, f"chronos_shadow_{target}.csv")
    if not os.path.exists(path):
        return None
    h = pd.read_csv(path, encoding="utf-8-sig")
    h["date"] = h["date"].astype(str).str[:10]
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            ["date", "hour_n"], keep="last")
    g = h[h["date"] == day]
    return g if len(g) == 24 else (g if not g.empty else None)


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 보조(shadow) 산출")
    ap.add_argument("--date", default=None,
                    help="대상일 YYYY-MM-DD (기본: 내일)")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju", "both"])
    args = ap.parse_args()
    day = args.date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    targets = (["smp_land", "smp_jeju"] if args.target == "both"
               else [args.target])

    print("=" * 70)
    print(" step8_chronos_shadow — Chronos 보조 산출")
    print(f" 대상일 {day} / blend_w={blend_weight(day)} / shadow={shadow_enabled()}")
    print("=" * 70)
    try:
        m = produce(day, targets=targets)
    except Exception as e:
        print(f"[생략] {type(e).__name__}: {e}")
        sys.exit(0)
    for t, df in m.items():
        print(f"  · {t}  구성={df['구성'].iloc[0]}  "
              f"평균={df['예측'].mean():.1f}  → chronos_shadow_{t}.csv")
    print("[완료]")


if __name__ == "__main__":
    main()
