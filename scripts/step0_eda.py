# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - Step 0-6 선행 데이터 분석(EDA)
================================================
목적 : 시간별 SMP의 구조적 패턴을 정량 규명하여 모델 해석·보고 근거를 확보
       (구축가이드 0단계 체크리스트 13~15번 충족)
입력 : data/processed/smp_features_{target}.csv
출력 : outputs/eda_{target}/*.png            (그래프 8종)
       outputs/eda_{target}/summary_*.csv    (분석표)
       logs/step0_eda_{target}_{일시}.csv    (주요 지표 요약)

분석 항목
  ① 기초 통계·연도별 추이        ② 시간대별 패턴(일유형·계절 교차)
  ③ 요일·휴일 효과               ④ 계절·월별 구조
  ⑤ 수요–가격 관계(탄력성)       ⑥ 신재생 침투율과 가격
  ⑦ 육지–제주 비교               ⑧ 자기상관·시차 구조

버전 : v1.0 (2026-08-04)
실행 : python scripts\\step0_eda.py
       python scripts\\step0_eda.py --target smp_jeju
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402
from step4_baseline import feat_paths  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

try:
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
except ImportError:
    sys.exit("[중단] matplotlib 미설치 → pip install matplotlib")

OUT_ROOT = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")

FONTS = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
         "AppleGothic", "Noto Sans CJK KR"]
KO = True
REPORT = []
C1, C2, C3 = "#1F3864", "#C00000", "#2E9E5B"


def setup_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["axes.unicode_minus"] = False
    for f in FONTS:
        if f in have:
            plt.rcParams["font.family"] = f
            return True, f
    return False, None


def L(ko, en):
    return ko if KO else en


SEASON_EN = {"봄": "Spring", "여름": "Summer", "가을": "Autumn", "겨울": "Winter"}
DAYTYPE_EN = {"평일": "Weekday", "휴일": "Holiday"}


def LB(v):
    """범주 라벨. 한글 폰트 미보유 시 영문 대체."""
    if KO:
        return str(v)
    return SEASON_EN.get(str(v), DAYTYPE_EN.get(str(v), str(v)))


def rep(구분, 항목, 값, 해석=""):
    REPORT.append({"구분": 구분, "항목": 항목, "값": str(값), "해석": 해석})
    print(f"  · {항목} : {값}" + (f"  ({해석})" if 해석 else ""))


# ------------------------------------------------------------------
def a1_basic(df, t, out):
    print("\n■ ① 기초 통계 및 연도별 추이")
    g = df.groupby("year")[t].agg(["mean", "std", "min", "max", "count"]).round(1)
    print(f"  {'연도':<6}{'평균':>9}{'표준편차':>10}{'최소':>9}{'최대':>9}{'N':>8}")
    for i, r in g.iterrows():
        print(f"  {int(i):<6}{r['mean']:>9.1f}{r['std']:>10.1f}"
              f"{r['min']:>9.1f}{r['max']:>9.1f}{int(r['count']):>8,}")
    g.to_csv(os.path.join(out, "summary_01_연도별.csv"), encoding="utf-8-sig")
    peak = g["mean"].idxmax()
    rep("기초", "최고 연도", f"{int(peak)}년 {g.loc[peak, 'mean']:.1f}원/kWh",
        "에너지 위기기" if int(peak) == 2022 else "")
    rep("기초", "변동성 최대 연도", f"{int(g['std'].idxmax())}년 σ={g['std'].max():.1f}")
    rep("기초", "전체 평균", f"{df[t].mean():.1f}원/kWh")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    m = df.groupby(df["date"].str[:7])[t].mean()
    ax.plot(pd.to_datetime(m.index + "-01"), m.values, lw=1.8, color=C1)
    ax.fill_between(pd.to_datetime(m.index + "-01"), m.values, alpha=.15, color=C1)
    ax.set_title(L("① 월평균 SMP 추이", "Monthly average SMP"), fontsize=12, weight="bold")
    ax.set_ylabel(L("원/kWh", "KRW/kWh")); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "01_월평균추이.png"), dpi=130)
    plt.close(fig)


def a2_hourly(df, t, out):
    print("\n■ ② 시간대별 패턴")
    g = df.groupby("hour_n")[t].mean()
    hi, lo = g.idxmax(), g.idxmin()
    rep("시간대", "최고 시간대", f"{hi}시 {g.max():.1f}원/kWh")
    rep("시간대", "최저 시간대", f"{lo}시 {g.min():.1f}원/kWh")
    rep("시간대", "일중 진폭", f"{g.max() - g.min():.1f}원/kWh",
        f"평균 대비 {(g.max() - g.min()) / g.mean() * 100:.1f}%")

    piv = df.pivot_table(index="hour_n", columns="season", values=t, aggfunc="mean")
    order = [s for s in ["봄", "여름", "가을", "겨울"] if s in piv.columns]
    piv = piv[order]
    piv.round(1).to_csv(os.path.join(out, "summary_02_시간대별.csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for c in piv.columns:
        ax[0].plot(piv.index, piv[c], lw=1.8, marker="o", ms=3, label=LB(c))
    ax[0].set_title(L("②-1 시간대별 계절 패턴", "Hourly by season"), fontsize=12, weight="bold")
    ax[0].set_xlabel(L("거래시간", "Hour")); ax[0].set_ylabel(L("원/kWh", "KRW/kWh"))
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_xticks(range(1, 25, 2))

    p2 = df.pivot_table(index="hour_n", columns="daytype", values=t, aggfunc="mean")
    for c, col in zip(p2.columns, [C1, C2]):
        ax[1].plot(p2.index, p2[c], lw=2, marker="o", ms=3, label=LB(c), color=col)
    ax[1].set_title(L("②-2 시간대별 일유형 패턴", "Hourly by daytype"), fontsize=12, weight="bold")
    ax[1].set_xlabel(L("거래시간", "Hour")); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_xticks(range(1, 25, 2))
    fig.tight_layout(); fig.savefig(os.path.join(out, "02_시간대별.png"), dpi=130)
    plt.close(fig)


def a3_daytype(df, t, out):
    print("\n■ ③ 요일·휴일 효과")
    dow_ko = ["월", "화", "수", "목", "금", "토", "일"]
    dow_lb = dow_ko if KO else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = df.groupby("dow")[t].mean()
    for i, v in g.items():
        print(f"  {dow_ko[int(i)]}요일 {v:>7.1f}원/kWh")
    wd = df[df["daytype"] == "평일"][t].mean()
    hd = df[df["daytype"] == "휴일"][t].mean()
    rep("일유형", "평일 평균", f"{wd:.1f}원/kWh")
    rep("일유형", "휴일 평균", f"{hd:.1f}원/kWh", f"평일 대비 {100 * (hd / wd - 1):+.1f}%")
    hol = df[df["is_holiday"] == 1][t].mean()
    rep("일유형", "공휴일 평균", f"{hol:.1f}원/kWh", f"평일 대비 {100 * (hol / wd - 1):+.1f}%")

    g2 = df.groupby("holiday_run_len")[t].agg(["mean", "size"])
    g2 = g2[g2.index > 0]
    if not g2.empty:
        rep("일유형", "연휴 길이별 최저",
            f"{int(g2['mean'].idxmin())}일 연휴 {g2['mean'].min():.1f}원/kWh")
    pd.DataFrame({"요일평균": g}).round(1).to_csv(
        os.path.join(out, "summary_03_요일별.csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].bar([dow_lb[int(i)] for i in g.index], g.values, color=C1, alpha=.85)
    ax[0].axhline(df[t].mean(), color=C2, ls="--", lw=1.2, label=L("전체 평균", "overall"))
    ax[0].set_title(L("③-1 요일별 평균 SMP", "By weekday"), fontsize=12, weight="bold")
    ax[0].set_ylabel(L("원/kWh", "KRW/kWh")); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

    if not g2.empty:
        ax[1].bar(g2.index.astype(int), g2["mean"], color=C3, alpha=.85)
        ax[1].axhline(wd, color=C2, ls="--", lw=1.2, label=L("평일 평균", "weekday"))
        ax[1].set_title(L("③-2 연휴 길이별 평균 SMP", "By holiday run"), fontsize=12, weight="bold")
        ax[1].set_xlabel(L("연휴 연속일수", "Run length")); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(out, "03_요일휴일.png"), dpi=130)
    plt.close(fig)


def a4_demand(df, t, out):
    print("\n■ ④ 수요–가격 관계")
    d = df.dropna(subset=["demand_fcst_total", t])
    r = d["demand_fcst_total"].corr(d[t])
    rep("수요", "수요예측–SMP 상관", f"{r:.3f}")

    # 로그-로그 회귀로 탄력성 추정
    m = d[(d[t] > 20) & (d["demand_fcst_total"] > 0)]
    b = np.polyfit(np.log(m["demand_fcst_total"]), np.log(m[t]), 1)[0]
    rep("수요", "가격의 수요 탄력성", f"{b:.2f}",
        "수요 1% 증가 시 SMP {:.2f}% 상승".format(b))

    d = d.copy()
    d["_구간"] = pd.qcut(d["demand_fcst_total"], 10, labels=False, duplicates="drop") + 1
    g = d.groupby("_구간").agg(수요=("demand_fcst_total", "mean"),
                              SMP=(t, "mean"), N=(t, "size")).round(1)
    g.to_csv(os.path.join(out, "summary_04_수요구간.csv"), encoding="utf-8-sig")
    print(f"  {'십분위':<8}{'평균수요(MW)':>14}{'평균SMP':>10}")
    for i, rr in g.iterrows():
        print(f"  {int(i):<8}{rr['수요']:>14,.0f}{rr['SMP']:>10.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    s = d.sample(min(8000, len(d)), random_state=0)
    ax[0].scatter(s["demand_fcst_total"], s[t], s=4, alpha=.15, color=C1, edgecolors="none")
    ax[0].set_title(L(f"④-1 수요예측 vs SMP (r={r:.3f})", f"Demand vs SMP (r={r:.3f})"),
                    fontsize=12, weight="bold")
    ax[0].set_xlabel(L("계통 수요예측 (MW)", "Demand (MW)"))
    ax[0].set_ylabel(L("원/kWh", "KRW/kWh")); ax[0].grid(alpha=.3)

    ax[1].plot(g["수요"], g["SMP"], marker="o", lw=2, color=C2)
    ax[1].set_title(L("④-2 수요 십분위별 평균 SMP", "SMP by demand decile"),
                    fontsize=12, weight="bold")
    ax[1].set_xlabel(L("평균 수요 (MW)", "Demand (MW)")); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "04_수요가격.png"), dpi=130)
    plt.close(fig)


def a5_renew(df, t, out):
    print("\n■ ⑤ 신재생 침투율과 가격")
    d = df.dropna(subset=["re_ratio", t])
    if d.empty or d["re_ratio"].std() == 0:
        print("  [건너뜀] 신재생 실적 데이터 부족")
        return
    r = d["re_ratio"].corr(d[t])
    rep("신재생", "침투율–SMP 상관", f"{r:.3f}",
        "침투율 상승 시 가격 하락" if r < 0 else "")
    d = d.copy()
    d["_구간"] = pd.qcut(d["re_ratio"], 10, labels=False, duplicates="drop") + 1
    g = d.groupby("_구간").agg(침투율=("re_ratio", "mean"), SMP=(t, "mean"),
                              N=(t, "size"))
    g["침투율"] = (g["침투율"] * 100).round(2)
    g["SMP"] = g["SMP"].round(1)
    g.to_csv(os.path.join(out, "summary_05_신재생.csv"), encoding="utf-8-sig")
    rep("신재생", "최저 침투율 구간 SMP", f"{g['SMP'].iloc[0]:.1f}원/kWh")
    rep("신재생", "최고 침투율 구간 SMP", f"{g['SMP'].iloc[-1]:.1f}원/kWh",
        f"차이 {g['SMP'].iloc[-1] - g['SMP'].iloc[0]:+.1f}원")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(g["침투율"], g["SMP"], marker="o", lw=2, color=C3)
    ax.set_title(L("⑤ 신재생 침투율 십분위별 평균 SMP", "SMP by RE penetration"),
                 fontsize=12, weight="bold")
    ax.set_xlabel(L("신재생 침투율 (%)", "RE ratio (%)"))
    ax.set_ylabel(L("원/kWh", "KRW/kWh")); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "05_신재생침투율.png"), dpi=130)
    plt.close(fig)


def a6_region(df, out):
    print("\n■ ⑥ 육지–제주 비교")
    d = df.dropna(subset=["smp_land", "smp_jeju"])
    diff = d["smp_jeju"] - d["smp_land"]
    rep("지역", "육지 평균", f"{d['smp_land'].mean():.1f}원/kWh")
    rep("지역", "제주 평균", f"{d['smp_jeju'].mean():.1f}원/kWh",
        f"육지 대비 {diff.mean():+.1f}원")
    rep("지역", "동일값 비율", f"{100 * (diff.abs() < 0.001).mean():.1f}%")
    rep("지역", "제주 변동성", f"σ={d['smp_jeju'].std():.1f}",
        f"육지 σ={d['smp_land'].std():.1f} 대비 {d['smp_jeju'].std() / d['smp_land'].std():.2f}배")

    g = d.groupby("hour_n").agg(육지=("smp_land", "mean"), 제주=("smp_jeju", "mean")).round(1)
    g["차이"] = (g["제주"] - g["육지"]).round(1)
    g.to_csv(os.path.join(out, "summary_06_지역비교.csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(g.index, g["육지"], lw=2, marker="o", ms=3, label=L("육지", "Land"), color=C1)
    ax[0].plot(g.index, g["제주"], lw=2, marker="s", ms=3, label=L("제주", "Jeju"), color=C2)
    ax[0].set_title(L("⑥-1 시간대별 육지·제주 SMP", "Land vs Jeju"), fontsize=12, weight="bold")
    ax[0].set_xlabel(L("거래시간", "Hour")); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xticks(range(1, 25, 2))

    ax[1].bar(g.index, g["차이"], color=C3, alpha=.85)
    ax[1].axhline(0, color="black", lw=.8)
    ax[1].set_title(L("⑥-2 시간대별 가격차(제주 - 육지)", "Jeju - Land"),
                    fontsize=12, weight="bold")
    ax[1].set_xlabel(L("거래시간", "Hour")); ax[1].set_ylabel(L("원/kWh", "KRW/kWh"))
    ax[1].grid(alpha=.3, axis="y"); ax[1].set_xticks(range(1, 25, 2))
    fig.tight_layout(); fig.savefig(os.path.join(out, "06_지역비교.png"), dpi=130)
    plt.close(fig)


def a7_autocorr(df, t, out):
    print("\n■ ⑦ 자기상관·시차 구조")
    s = df.sort_values(["date", "hour_n"])[t].reset_index(drop=True)
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720]
    ac = {l: s.autocorr(l) for l in lags}
    print(f"  {'시차(h)':<10}{'자기상관':>10}{'비고':>16}")
    note = {24: "전일 동시간", 168: "전주 동시간", 720: "전월 동시간"}
    rows = []
    for l, v in ac.items():
        print(f"  {l:<10}{v:>10.3f}{note.get(l, ''):>16}")
        rows.append({"시차(h)": l, "자기상관": round(v, 4), "비고": note.get(l, "")})
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary_07_자기상관.csv"),
                              index=False, encoding="utf-8-sig")
    rep("시차", "lag24 자기상관", f"{ac[24]:.3f}", "전일 동시간")
    rep("시차", "lag168 자기상관", f"{ac[168]:.3f}", "전주 동시간")
    rep("시차", "지속성 반감 시차",
        f"{min([l for l in lags if ac[l] < ac[1] / 2], default='-')}h")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(range(len(lags)), [ac[l] for l in lags], color=C1, alpha=.85)
    ax.set_xticks(range(len(lags))); ax.set_xticklabels(lags)
    ax.axhline(0, color="black", lw=.8)
    ax.set_title(L("⑦ 시차별 자기상관", "Autocorrelation by lag"), fontsize=12, weight="bold")
    ax.set_xlabel(L("시차 (시간)", "Lag (hours)")); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(out, "07_자기상관.png"), dpi=130)
    plt.close(fig)


def a8_corr(df, t, out):
    print("\n■ ⑧ 주요 변수 상관")
    cand = ["demand_fcst_total", "demand_fcst_land", "demand_fcst_jeju",
            "demand_fcst_ratio", "demand_actual", "pv_total", "wind_total",
            "net_load", "re_ratio", "hour_n", "is_holiday", "is_weekend"]
    cols = [c for c in cand if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    c = df[cols + [t]].corr()[t].drop(t).sort_values(key=abs, ascending=False)
    for k, v in c.items():
        print(f"  {k:<24}{v:>8.3f}")
    c.round(4).to_csv(os.path.join(out, "summary_08_상관.csv"), encoding="utf-8-sig")
    rep("상관", "최대 상관 변수", f"{c.index[0]} ({c.iloc[0]:.3f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    cc = c.sort_values()
    ax.barh(range(len(cc)), cc.values,
            color=[C2 if v < 0 else C1 for v in cc.values], alpha=.85)
    ax.set_yticks(range(len(cc))); ax.set_yticklabels(cc.index, fontsize=8)
    ax.axvline(0, color="black", lw=.8)
    ax.set_title(L("⑧ 주요 변수와 SMP 상관계수", "Correlation with SMP"),
                 fontsize=12, weight="bold")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(out, "08_상관계수.png"), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    global KO
    ap = argparse.ArgumentParser(description="Step 0-6 선행 데이터 분석")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    args = ap.parse_args()
    t = args.target

    SRC, _ = feat_paths(t)
    if not os.path.exists(SRC):
        sys.exit(f"[중단] 입력파일 없음 → {SRC}")
    out = os.path.join(OUT_ROOT, f"eda_{t}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ok, fname = setup_font()
    KO = ok

    print("=" * 74)
    print(" Step 0-6 선행 데이터 분석 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print(f" 한글폰트 : {fname if ok else '미보유 → 영문 라벨'}")
    print("=" * 74)

    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    print(f"\n 분석 대상 : {len(df):,}행 / {df['date'].min()} ~ {df['date'].max()}")

    a1_basic(df, t, out)
    a2_hourly(df, t, out)
    a3_daytype(df, t, out)
    a4_demand(df, t, out)
    a5_renew(df, t, out)
    a6_region(df, out)
    a7_autocorr(df, t, out)
    a8_corr(df, t, out)

    lp = os.path.join(LOG_DIR, f"step0_eda_{t}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    pd.DataFrame(REPORT).to_csv(lp, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 74)
    print(f" 그래프·분석표 : {out}")
    print(f" 지표 요약     : {lp}")
    print(" ※ 본 결과는 구축가이드 0단계 체크리스트 13~15번 근거자료로 활용")
    print("=" * 74)


if __name__ == "__main__":
    main()
