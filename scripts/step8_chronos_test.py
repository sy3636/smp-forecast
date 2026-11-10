# -*- coding: utf-8 -*-
"""
step8_chronos_test.py — Chronos-2 제로샷 D+1 실증 (v1.0)
==============================================================================
배경
  · 숙명여대 권영옥 교수 AI 심화교육에서 Chronos-2(아마존 시계열 기반모델) 우수성 소개
  · 본 프로그램(LightGBM 앙상블 + 편의보정)과 **동일 지표·동일 누출 규칙**으로 대조 필요

모델
  · amazon/chronos-2 (120M) — 기본
  · autogluon/chronos-2-small (28M) — CPU 파일럿용 (--model)
  · 패키지: chronos-forecasting>=2.0 (운영 requirements 미포함 — 선택 설치)

설계
  · 대상: 육지 시간별 SMP D+1 (24시간)
  · walk-forward: 대상일 직전 시각까지만 context — **누출 차단**
  · 구성
      ① Chronos-2 단변량(uni)
      ② Chronos-2 + 수요예측 공변량(cov) — demand_fcst_land를 known future로 투입
         (운영 21:00 이후 산출과 동일 가용성 가정)
  · 대조: naive-1(전일 동시간) · 운영 이력(predictions_*.csv, 있으면)

판정 지표 (PRD v2.13)
  · MAE · rMAE(÷naive-1) · MAPE(분모하한 20) · 편의

유의
  · 운영 모델을 갱신하지 않음(실증 전용)
  · CPU 추론은 느리므로 --start/--end 로 구간을 제한할 것
  · 첫 실행 시 Hugging Face에서 가중치를 내려받음(네트워크 필요)

실행
  pip install -r requirements-chronos.txt
  python scripts\\step8_chronos_test.py
  python scripts\\step8_chronos_test.py --start 2026-08-01 --end 2026-09-13
  python scripts\\step8_chronos_test.py --model amazon/chronos-2 --mode both
  python scripts\\step8_chronos_test.py --model autogluon/chronos-2-small --device cpu

버전 : v1.0 (2026-09-12)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "data", "processed", "smp_dataset_hourly.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")

FLOOR = 20.0
CTX_DEFAULT = 2048          # 시간 — Chronos-2 최대 8192, CPU 균형용
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<28} {str(값):<28} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def mape(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    if not m.any():
        return np.nan
    return float((np.abs(p[m] - a[m]) / np.maximum(a[m], FLOOR)).mean() * 100)


def bias(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    if not m.any():
        return np.nan
    return float(((p[m] - a[m]) / np.maximum(a[m], FLOOR)).mean() * 100)


def metrics(p, a, naive):
    m = mae(p, a)
    n = mae(naive, a)
    return {
        "MAE": round(m, 2),
        "rMAE": round(m / n, 3) if n and n > 0 else np.nan,
        "MAPE": round(mape(p, a), 2),
        "편의": round(bias(p, a), 2),
        "naive_MAE": round(n, 2),
        "개선율": round((1 - m / n) * 100, 1) if n and n > 0 else np.nan,
    }


def load_series(target):
    usecols = ["date", "hour_n", target]
    raw = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False)
    raw.columns = [c.lstrip("\ufeff") for c in raw.columns]
    cov = "demand_fcst_land" if target == "smp_land" else "demand_fcst_jeju"
    if cov in raw.columns:
        usecols.append(cov)
    d = raw[usecols].copy()
    d["date"] = d["date"].astype(str).str[:10]
    d = d.dropna(subset=[target]).sort_values(["date", "hour_n"]).reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["date"]) + pd.to_timedelta(d["hour_n"] - 1, unit="h")
    d["naive"] = d[target].shift(24)
    return d, cov if cov in d.columns else None


def load_ops(target):
    p = os.path.join(OUT_DIR, f"predictions_{target}.csv")
    if not os.path.exists(p):
        return None
    h = pd.read_csv(p, encoding="utf-8-sig")
    h.columns = [c.lstrip("\ufeff") for c in h.columns]
    # 최신 예측만 유지
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            ["date", "hour_n"], keep="last")
    h["date"] = h["date"].astype(str).str[:10]
    h = h.dropna(subset=["예측", "실적"])
    return h[["date", "hour_n", "예측", "실적"]].copy()


def build_pipeline(model_id, device):
    from chronos import Chronos2Pipeline
    print(f"\n■ 모델 적재 : {model_id} / device={device}", flush=True)
    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
    print(f"  · 적재 소요 {time.time() - t0:.1f}초", flush=True)
    return pipe


def forecast_day(pipe, hist, day, target, cov_col, mode, context_length):
    fut = hist[hist["date"] == day].copy()
    if len(fut) != 24:
        return None, f"대상일 {len(fut)}시간"
    cutoff = fut["timestamp"].iloc[0]
    ctx = hist[hist["timestamp"] < cutoff].tail(context_length)
    if len(ctx) < 48:
        return None, f"context 부족({len(ctx)})"

    context_df = pd.DataFrame({
        "item_id": "smp",
        "timestamp": ctx["timestamp"].values,
        "target": ctx[target].astype(float).values,
    })
    future_df = None
    if mode == "cov":
        if cov_col is None or fut[cov_col].isna().any():
            return None, "공변량 결측"
        context_df[cov_col] = ctx[cov_col].astype(float).values
        future_df = pd.DataFrame({
            "item_id": "smp",
            "timestamp": fut["timestamp"].values,
            cov_col: fut[cov_col].astype(float).values,
        })

    pred = pipe.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=24,
        quantile_levels=[0.5],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        context_length=context_length,
        validate_inputs=True,
    )
    cols = [c for c in pred.columns if c not in ("item_id", "timestamp")]
    q = "0.5" if "0.5" in pred.columns else (
        "predictions" if "predictions" in pred.columns else cols[-1])
    out = fut[["date", "hour_n", target, "naive"]].copy()
    out = out.rename(columns={target: "실적"})
    pred = pred.sort_values("timestamp").reset_index(drop=True)
    out = out.reset_index(drop=True)
    out["예측"] = pred[q].astype(float).values[:24]
    out["구성"] = mode
    return out, "ok"


def summarize(tag, df):
    if df is None or df.empty:
        return None
    m = metrics(df["예측"], df["실적"], df["naive"])
    print(f"\n■ {tag}")
    for k, v in m.items():
        rep(f"{tag} {k}", v, f"N={len(df)}")
    return m


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 D+1 제로샷 실증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--model", default="amazon/chronos-2",
                    help="HF model id (예: autogluon/chronos-2-small)")
    ap.add_argument("--mode", default="both",
                    choices=["uni", "cov", "both"])
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--end", default=None, help="포함 종료일(기본: 데이터 최신 전일)")
    ap.add_argument("--context", type=int, default=CTX_DEFAULT)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    args = ap.parse_args()

    print("=" * 78)
    print(" step8_chronos_test — Chronos-2 제로샷 D+1 실증 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    if not os.path.exists(DATASET):
        sys.exit(f"[중단] 데이터 없음 → {DATASET}")

    hist, cov_col = load_series(args.target)
    days_all = sorted(hist["date"].unique())
    end = args.end or days_all[-2]
    eval_days = [d for d in days_all if args.start <= d <= end]
    if not eval_days:
        sys.exit("[중단] 평가 일자 없음")

    print("\n■ ① 대상")
    rep("타깃", args.target, "")
    rep("모델", args.model, args.device)
    rep("평가 구간", f"{eval_days[0]} ~ {eval_days[-1]}", f"{len(eval_days)}일")
    rep("context", f"{args.context}시간", f"약 {args.context/24:.0f}일")
    rep("공변량", cov_col or "없음", "cov 모드")

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    pipe = build_pipeline(args.model, device)
    modes = ["uni", "cov"] if args.mode == "both" else [args.mode]
    if "cov" in modes and not cov_col:
        print("  · cov 모드 제외 — 수요예측 열 없음", flush=True)
        modes = [m for m in modes if m != "cov"]

    frames = {m: [] for m in modes}
    t_all = time.time()
    for i, day in enumerate(eval_days, 1):
        for mode in modes:
            t0 = time.time()
            try:
                out, why = forecast_day(
                    pipe, hist, day, args.target, cov_col, mode, args.context)
            except Exception as e:
                out, why = None, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            if out is None:
                print(f"  [{i}/{len(eval_days)}] {day} {mode} 실패 — {why}",
                      flush=True)
                continue
            frames[mode].append(out)
            err = (out["예측"] - out["실적"]).abs().mean()
            print(f"  [{i}/{len(eval_days)}] {day} {mode}  "
                  f"MAE {err:.2f}  ({dt:.1f}초)", flush=True)

    print(f"\n■ ② 종합 (소요 {time.time() - t_all:.0f}초)")
    results = {}
    detail_rows = []
    for mode, parts in frames.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        results[mode] = summarize(f"Chronos-{mode}", df)
        detail_rows.append(df)

    base = hist[hist["date"].isin(eval_days)].dropna(subset=["naive"])
    if not base.empty:
        results["naive"] = summarize(
            "naive-1",
            base.assign(예측=base["naive"], 실적=base[args.target]))

    ops = load_ops(args.target)
    if ops is not None:
        o = ops[ops["date"].isin(eval_days)].copy()
        if not o.empty:
            o = o.merge(
                hist[["date", "hour_n", "naive"]],
                on=["date", "hour_n"], how="left")
            results["운영"] = summarize("운영(LightGBM+보정)", o)

    print("\n■ ③ 판정")
    cu = results.get("uni")
    cc = results.get("cov")
    op = results.get("운영")
    nv = results.get("naive")

    def line(name, m):
        if not m:
            return
        print(f"    · {name}: MAE {m['MAE']} / rMAE {m['rMAE']} / "
              f"MAPE {m['MAPE']}% / 편의 {m['편의']:+}% "
              f"(naive대비 개선 {m['개선율']}%)")

    line("Chronos uni", cu)
    line("Chronos cov", cc)
    line("운영 이력", op)
    line("naive-1", nv)

    cands = [(f"Chronos-{k}", v) for k, v in results.items()
             if k in ("uni", "cov") and v and not np.isnan(v["MAE"])]
    if cands:
        best = min(cands, key=lambda x: x[1]["MAE"])
        print(f"    · Chronos 최선: {best[0]} (MAE {best[1]['MAE']})")
        if op and best[1]["MAE"] < op["MAE"]:
            print("    · 운영 대비 우위 — 채택 검토(운영 부하·의존성 별도)")
            rep("판정", "운영 대비 우위", best[0])
        elif op:
            print("    · 운영 대비 열위 또는 동등 — 현행 유지, Chronos는 보조 후보")
            rep("판정", "현행 유지(보조 후보)",
                f"운영 MAE {op['MAE']} vs {best[0]} {best[1]['MAE']}")
        else:
            rep("판정", "운영 이력 없어 Chronos 단독 측정", best[0])

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_chronos_test_{args.target}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        pd.DataFrame(_rows).to_csv(f, index=False)
        f.write("\n")
        if detail_rows:
            pd.concat(detail_rows, ignore_index=True).to_csv(f, index=False)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
