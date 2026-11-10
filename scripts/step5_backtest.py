# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step5 백테스트·시각화 리포트
================================================
목적 : 실운영 조건(주기적 재학습)을 재현한 백테스트를 수행하고,
       보고용 시각화·성능표를 자동 산출
입력 : data/processed/smp_features_{target}.csv, feature_dictionary_{target}.csv
       data/processed/baseline_predictions_{target}.csv
출력 : outputs/backtest_{target}/*.png            (그래프 5종)
       outputs/backtest_{target}/predictions.csv  (예측 원본)
       logs/step5_backtest_{target}_{일시}.csv    (성능·오차분해)

백테스트 설계
  · 재학습 주기 : 월 1회(--refit monthly) — 매월 초 직전월까지 데이터로 재학습
                  none 지정 시 2025-12-31 시점 1회 학습분으로 전 구간 예측
  · 예측 시점   : D-1 23시(익일 발전계획 확정 직후)
  · 평가 구간   : 2026-01-01 ~ 2026-07-30 (홀드아웃)
  · MAPE 하한   : 20원/kWh (step4_baseline.MAPE_FLOOR)

버전 : v1.0 (2026-08-01)
실행 : python scripts\\step5_backtest.py
       python scripts\\step5_backtest.py --target smp_jeju
       python scripts\\step5_backtest.py --refit none
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import (MAPE_FLOOR, mape_floor, set_mape_floor,  # noqa: E402
                            feat_paths, metrics, TRAIN_START, CV_END, HOLD_START)
from step4_train import (EXCLUDE, CATEGORICAL, load_model_class,  # noqa: E402
                         load_tuned_params, make_model, base_pred_path)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[중단] matplotlib 미설치 → pip install matplotlib")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
OUT_ROOT = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 한글 폰트 우선순위(Windows·macOS·Linux)
FONTS = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
         "AppleGothic", "Noto Sans CJK KR"]


def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def setup_font():
    """한글 폰트 설정. 미보유 시 영문 라벨로 대체."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for f in FONTS:
        if f in have:
            plt.rcParams["font.family"] = f
            plt.rcParams["axes.unicode_minus"] = False
            return True, f
    plt.rcParams["axes.unicode_minus"] = False
    return False, None


KO = True   # 한글 라벨 사용 여부(폰트 확인 후 갱신)


def L(ko, en):
    return ko if KO else en


# ------------------------------------------------------------------
# 백테스트
# ------------------------------------------------------------------
def run_backtest(df, feats, t, kind, lgb, tuned, seed, refit):
    """월별 재학습 롤링 예측. refit='none' 이면 1회 학습분으로 전 구간 예측."""
    ho = df[df["date"] >= HOLD_START].copy()
    ho["pred"] = np.nan
    ho["학습기준일"] = ""
    fits = []

    if refit == "none":
        tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= CV_END)].dropna(subset=[t])
        m = make_model(kind, lgb, seed, tuned)
        m.fit(tr[feats], tr[t])
        ho["pred"] = m.predict(ho[feats])
        ho["학습기준일"] = CV_END
        fits.append((CV_END, len(tr), len(ho)))
        return ho, fits

    months = sorted(ho["date"].str[:7].unique())
    for ym in months:
        cut = (pd.Timestamp(ym + "-01") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= cut)].dropna(subset=[t])
        seg = ho["date"].str[:7] == ym
        m = make_model(kind, lgb, seed, tuned)
        m.fit(tr[feats], tr[t])
        ho.loc[seg, "pred"] = m.predict(ho.loc[seg, feats])
        ho.loc[seg, "학습기준일"] = cut
        fits.append((cut, len(tr), int(seg.sum())))
    return ho, fits


# ------------------------------------------------------------------
# 시각화
# ------------------------------------------------------------------
def fig_timeseries(ho, t, path):
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    d = ho.copy()
    d["ts"] = pd.to_datetime(d["date"]) + pd.to_timedelta(d["hour_n"] - 1, unit="h")
    ax[0].plot(d["ts"], d[t], lw=0.7, label=L("실적", "Actual"), color="#1F3864")
    ax[0].plot(d["ts"], d["pred"], lw=0.7, label=L("예측", "Forecast"), color="#C00000", alpha=.8)
    ax[0].set_title(L("① 전 구간 시계열 (2026.1~7)", "Full period"), fontsize=12, weight="bold")
    ax[0].set_ylabel(L("SMP (원/kWh)", "SMP (KRW/kWh)"))
    ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)

    last = d[d["ts"] >= d["ts"].max() - pd.Timedelta(days=21)]
    ax[1].plot(last["ts"], last[t], lw=1.2, label=L("실적", "Actual"), color="#1F3864")
    ax[1].plot(last["ts"], last["pred"], lw=1.2, label=L("예측", "Forecast"), color="#C00000")
    ax[1].fill_between(last["ts"], last[t], last["pred"], alpha=.15, color="#C00000")
    ax[1].set_title(L("② 최근 3주 확대", "Last 3 weeks"), fontsize=12, weight="bold")
    ax[1].set_ylabel(L("SMP (원/kWh)", "SMP (KRW/kWh)"))
    ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_scatter(ho, t, path):
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(ho[t], ho["pred"], s=6, alpha=.25, color="#1F3864", edgecolors="none")
    lo = float(min(ho[t].min(), ho["pred"].min()))
    hi = float(max(ho[t].max(), ho["pred"].max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label=L("완전일치선", "y = x"))
    r2 = metrics(ho[t], ho["pred"])["R2"]
    ax.set_xlabel(L("실적 SMP (원/kWh)", "Actual")); ax.set_ylabel(L("예측 SMP (원/kWh)", "Forecast"))
    ax.set_title(L(f"③ 실적–예측 산점도 (R² = {r2:.3f})", f"Scatter (R2={r2:.3f})"),
                 fontsize=12, weight="bold")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_error(ho, t, path):
    err = ho["pred"] - ho[t]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(err, bins=60, color="#2E5984", alpha=.85)
    ax[0].axvline(0, color="r", ls="--", lw=1)
    ax[0].axvline(err.mean(), color="orange", lw=1.5,
                  label=L(f"평균 {err.mean():+.2f}", f"mean {err.mean():+.2f}"))
    ax[0].set_title(L("④-1 오차 분포", "Error distribution"), fontsize=12, weight="bold")
    ax[0].set_xlabel(L("예측 - 실적 (원/kWh)", "Forecast - Actual")); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].scatter(ho[t], err, s=6, alpha=.25, color="#2E5984", edgecolors="none")
    ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set_title(L("④-2 잔차 진단(실적 대비)", "Residual vs actual"), fontsize=12, weight="bold")
    ax[1].set_xlabel(L("실적 SMP (원/kWh)", "Actual")); ax[1].set_ylabel(L("오차", "Error"))
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_heatmap(ho, path):
    piv = ho.pivot_table(index="hour_n", columns=ho["date"].str[:7],
                         values="_ape", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45)
    ax.set_yticks(range(0, 24, 2)); ax.set_yticklabels(range(1, 25, 2))
    ax.set_xlabel(L("월", "Month")); ax.set_ylabel(L("거래시간", "Hour"))
    ax.set_title(L("⑤ 시간대 × 월별 MAPE 히트맵 (%)", "MAPE heatmap"), fontsize=12, weight="bold")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanpercentile(piv.values, 75) else "black")
    fig.colorbar(im, ax=ax, label="MAPE (%)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_daily(ho, path):
    d = ho.groupby("date")["_ape"].mean().reset_index()
    d["dt"] = pd.to_datetime(d["date"])
    d["ma7"] = d["_ape"].rolling(7, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(d["dt"], d["_ape"], width=.9, color="#8FAADC", label=L("일별 MAPE", "Daily MAPE"))
    ax.plot(d["dt"], d["ma7"], color="#C00000", lw=1.8, label=L("7일 이동평균", "7d MA"))
    ax.axhline(8.0, color="green", ls="--", lw=1.2, label=L("Gate 3 기준 8.0%", "Gate 3 = 8.0%"))
    ax.set_ylabel("MAPE (%)"); ax.grid(alpha=.3); ax.legend(loc="upper right")
    ax.set_title(L("⑥ 일별 MAPE 추이", "Daily MAPE trend"), fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------
def main():
    global KO
    ap = argparse.ArgumentParser(description="step5 백테스트·시각화")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--refit", default="monthly", choices=["monthly", "none"],
                    help="재학습 주기(기본 monthly)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR)
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)
    t = args.target

    SRC, DICT = feat_paths(t)
    for p in (SRC, DICT):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}")
    out_dir = os.path.join(OUT_ROOT, f"backtest_{t}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ok_font, fname = setup_font()
    KO = ok_font
    kind, lgb = load_model_class()
    tuned, src_p = load_tuned_params(kind, t)

    print("=" * 78)
    print(" step5 백테스트·시각화 리포트 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" 타깃 : {t} / 엔진 : {kind} / 파라미터 : {src_p} / 재학습 : {args.refit}")
    print(f" 한글폰트 : {fname if ok_font else '미보유 → 영문 라벨 사용'}")
    print("=" * 78)

    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(DICT, encoding="utf-8-sig")
    okv = set(dd[dd["운영가용"] == "Y"]["변수명"])
    feats = [c for c in df.columns if c in okv and c not in EXCLUDE]
    cats = [c for c in CATEGORICAL if c in feats]
    for c in cats:
        df[c] = df[c].astype("category")
    feats = [c for c in feats if c in cats or is_num(df[c])]

    # ── ① 백테스트 실행 ────────────────────────
    print(f"\n■ ① 백테스트 실행 (투입 변수 {len(feats)}개)")
    ho, fits = run_backtest(df, feats, t, kind, lgb, tuned, args.seed, args.refit)
    print(f"  {'학습기준일':<14}{'학습행수':>10}{'예측행수':>10}")
    for c, n1, n2 in fits:
        print(f"  {c:<14}{n1:>10,}{n2:>10,}")

    ho["_err"] = ho["pred"] - ho[t]
    ho["_ape"] = (ho["_err"] / ho[t].where(ho[t].abs() >= mape_floor())).abs() * 100

    # ── ② 성능 ─────────────────────────────────
    print("\n■ ② 백테스트 성능")
    mm = metrics(ho[t], ho["pred"])
    print(f"  {'MAPE':>8}{'sMAPE':>8}{'RMSE':>9}{'MAE':>8}{'R²':>8}{'N':>8}")
    print(f"  {mm['MAPE']:>8.2f}{mm['sMAPE']:>8.2f}{mm['RMSE']:>9.2f}"
          f"{mm['MAE']:>8.2f}{mm['R2']:>8.3f}{mm['N']:>8,}")
    gate = "달성" if mm["MAPE"] <= 8.0 else "미달"
    print(f"  → Gate 3 기준(MAPE ≤ 8.0%) : {gate}")

    rows = [{"구분": "종합", "항목": f"{args.refit} 재학습", "MAPE": round(mm["MAPE"], 3),
             "RMSE": round(mm["RMSE"], 2), "MAE": round(mm["MAE"], 2),
             "R2": round(mm["R2"], 3), "N": mm["N"]}]

    # 벤치마크 대비
    bpp = base_pred_path(t)
    if os.path.exists(bpp):
        bp = pd.read_csv(bpp, encoding="utf-8-sig")
        bp = bp[bp["date"] >= HOLD_START]
        mg = ho[["date", "hour_n", t]].merge(bp, on=["date", "hour_n"], how="left")
        y = mg[f"{t}_x"] if f"{t}_x" in mg.columns else mg[t]
        cand = {c: metrics(y, mg[c]) for c in bp.columns if c.startswith("B")}
        best = min(cand, key=lambda k: cand[k]["MAPE"])
        imp = 100 * (cand[best]["MAPE"] - mm["MAPE"]) / cand[best]["MAPE"]
        print(f"  → 최우수 벤치마크({best}) {cand[best]['MAPE']:.2f}% 대비 {imp:+.1f}% 개선")
        rows.append({"구분": "벤치마크", "항목": best, "MAPE": round(cand[best]["MAPE"], 3),
                     "RMSE": round(cand[best]["RMSE"], 2), "MAE": round(cand[best]["MAE"], 2),
                     "R2": round(cand[best]["R2"], 3), "N": cand[best]["N"]})

    # ── ③ 오차 분해 ────────────────────────────
    print("\n■ ③ 오차 분해")
    ho["_구간"] = pd.cut(ho[t], [-999, 20, 80, 100, 120, 150, 9999],
                        labels=["20원미만", "20~80", "80~100", "100~120", "120~150", "150원~"])
    for label, key in (("시간대별", "hour_n"), ("월별", ho["date"].str[:7].rename("월")),
                       ("일유형별", "daytype"), ("SMP구간별", "_구간")):
        g = ho.groupby(key, observed=True).agg(
            MAPE=("_ape", "mean"),
            RMSE=("_err", lambda x: float(np.sqrt((x ** 2).mean()))),
             편의=("_err", "mean"), N=("_err", "size"))
        print(f"\n  [{label}]")
        for i, r in g.iterrows():
            print(f"  {str(i):<10}MAPE {r['MAPE']:>7.2f}%  RMSE {r['RMSE']:>6.2f}  "
                  f"편의 {r['편의']:>+6.2f}  N {int(r['N']):>6,}")
            rows.append({"구분": label, "항목": str(i), "MAPE": round(r["MAPE"], 2),
                         "RMSE": round(r["RMSE"], 2), "MAE": None,
                         "R2": None, "N": int(r["N"]), "편의": round(r["편의"], 2)})

    # ── ④ 시각화 ───────────────────────────────
    print("\n■ ④ 시각화 생성")
    figs = [("01_시계열.png", fig_timeseries, (ho, t)),
            ("02_산점도.png", fig_scatter, (ho, t)),
            ("03_오차분포.png", fig_error, (ho, t)),
            ("04_히트맵.png", fig_heatmap, (ho,)),
            ("05_일별추이.png", fig_daily, (ho,))]
    for name, fn, a in figs:
        p = os.path.join(out_dir, name)
        fn(*a, p)
        print(f"  · {name}")

    # ── ⑤ 저장 ─────────────────────────────────
    ho[["date", "hour_n", "학습기준일", t, "pred", "_err", "_ape"]].rename(
        columns={"_err": "오차", "_ape": "APE"}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")
    lp = os.path.join(LOG_DIR,
                      f"step5_backtest_{t}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    pd.DataFrame(rows).to_csv(lp, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f" 산출물 : {out_dir}")
    print(f" 성능표 : {lp}")
    print(f" 결론   : {args.refit} 재학습 기준 MAPE {mm['MAPE']:.2f}% / "
          f"RMSE {mm['RMSE']:.2f}원-kWh / Gate 3 {gate}")
    print("=" * 78)


if __name__ == "__main__":
    main()
