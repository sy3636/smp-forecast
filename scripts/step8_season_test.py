# -*- coding: utf-8 -*-
"""
step8_season_test.py — 계절 전환기 과대예측 검증 (v1.0)
==============================================================================
배경
  · 2026. 9. 7.(월) 주간(10~16시) 예측이 실적 대비 과대하다는 관측 제기
  · 가설 : 가을 진입에 따른 **냉방수요 감소**가 예측에 반영되지 않음

구조적 근거 — 기온계 변수의 모델 미투입
  · 변수사전 확인 결과 다음 변수가 전량 **운영가용 = N**(모델 미투입, 분석 전용)
    temp_land · temp_jeju · cdd_land · hdd_land · temp_ma24 · temp_ma72 ·
    temp_daily_max · temp_diff_d1 등
  · 사유 : 2026. 8. 6. 「기상 변수 25종」 실증에서 개선 ±0.00%p로 기각
           — KPX 하루전 수요예측값에 기상 전망이 이미 내재된다는 판단
  · 다만 동 실증은 **연중 전체** 기준이며, **계절 전환기 한정 효과는 미검증**

점검 항목
  ① 대상일 시간대별 예측·실적·오차 — 과대예측 구간 특정
  ② 계절 전환 규모 — 연도별 8월 하순 대비 9월 상순 주간 SMP 변화
  ③ 최근 편의의 시간대 구조 — 주간 편중 여부
  ④ 월별 주간(10~16시) 편의 추이 — 계절성 확인

실증(--ab)
  · 기온계 변수 4종(cdd_land · temp_ma24 · temp_daily_max · temp_diff_d1) 투입 A/B
  · **운영과 동일한 7일 주기 재학습** 조건에서 평가
  · 전체·9월 구간·주간(10~16시) 구간으로 분해하여 판정

실행
  python scripts\\step8_season_test.py --date 2026-09-07
  python scripts\\step8_season_test.py --ab --start 2026-06-01

버전 : v1.0 (2026-09-07)
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET = os.path.join(PROCESSED, "smp_dataset_hourly.csv")

MAPE_FLOOR = 20.0
SEED = 42
TRAIN_START = "2021-02-01"
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju", "year",
           "unit_id", "unit_name", "grp", "pv_source", "holiday_name"}
TEMP_VARS = ["cdd_land", "temp_ma24", "temp_daily_max", "temp_diff_d1"]
DAY_HOURS = list(range(10, 17))          # 주간 10~16시
_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {str(값):<24} {비고}", flush=True)
    _log.append({"항목": 항목, "값": str(값), "비고": 비고})


def mae(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(np.abs(p[m] - a[m]).mean()) if m.any() else np.nan


def bias(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    m = ~(np.isnan(p) | np.isnan(a))
    return float(((p[m] - a[m]) / np.maximum(a[m], MAPE_FLOOR)).mean() * 100)


def load_hist(t):
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 운영 이력 없음 → {p}")
    h = pd.read_csv(p, encoding="utf-8-sig").dropna(subset=["예측", "실적"])
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    h["date"] = h["date"].astype(str)
    return h.sort_values(["date", "hour_n"]).reset_index(drop=True)


def diagnose(t, target_date):
    h = load_hist(t)
    ds = load_dataset(t)

    # ── ① 대상일 시간대별 ──────────────────────
    seg = h[h["date"] == target_date]
    print(f"\n■ ① 대상일 {target_date} 시간대별 오차")
    if seg.empty:
        print("  [주의] 해당 일자 이력 없음 — 실적 미확정 또는 산출 부재")
    else:
        print(f"  {'시':>3}{'예측':>9}{'실적':>9}{'오차':>9}{'기온':>7}")
        print("  " + "-" * 38)
        tmp = ds.set_index(["date", "hour_n"])["temp_land"] \
            if "temp_land" in ds.columns else None
        for _, r in seg.iterrows():
            tv = ""
            if tmp is not None:
                try:
                    tv = f"{float(tmp.loc[(r['date'], r['hour_n'])]):.1f}"
                except Exception:
                    tv = "-"
            print(f"  {int(r['hour_n']):>3}{r['예측']:>9.2f}{r['실적']:>9.2f}"
                  f"{r['예측'] - r['실적']:>9.2f}{tv:>7}")
        print("  " + "-" * 38)
        d = seg[seg["hour_n"].isin(DAY_HOURS)]
        n = seg[~seg["hour_n"].isin(DAY_HOURS)]
        rep("주간(10~16시)", f"MAE {mae(d['예측'], d['실적']):.2f}원",
            f"편의 {bias(d['예측'], d['실적']):+.2f}%")
        rep("그 외 시간대", f"MAE {mae(n['예측'], n['실적']):.2f}원",
            f"편의 {bias(n['예측'], n['실적']):+.2f}%")

    # ── ② 계절 전환 규모(연도별) ────────────────
    print("\n■ ② 계절 전환 규모 — 8월 하순 대비 9월 상순 주간(10~16시) 실적")
    ds["ym"] = ds["date"].str[:7]
    ds["dd"] = ds["date"].str[8:10].astype(int)
    day = ds[ds["hour_n"].isin(DAY_HOURS)]
    print(f"  {'연도':<6}{'8월하순':>10}{'9월상순':>10}{'변화':>10}{'변화율':>9}")
    print("  " + "-" * 46)
    for y in sorted({x[:4] for x in ds["ym"].unique()}):
        a = day[(day["ym"] == f"{y}-08") & (day["dd"] >= 21)][t].mean()
        b = day[(day["ym"] == f"{y}-09") & (day["dd"] <= 10)][t].mean()
        if np.isnan(a) or np.isnan(b):
            continue
        print(f"  {y:<6}{a:>10.2f}{b:>10.2f}{b - a:>10.2f}{(b / a - 1) * 100:>8.1f}%")
        rep(f"전환 {y}", f"{a:.1f} → {b:.1f}원", f"{(b / a - 1) * 100:+.1f}%")
    print("  " + "-" * 46)

    # ── ③ 최근 편의의 시간대 구조 ───────────────
    print("\n■ ③ 최근 14일 시간대별 편의(운영 이력)")
    days = sorted(h["date"].unique())[-14:]
    s = h[h["date"].isin(days)]
    print(f"  {'시':>3}{'MAE':>9}{'편의%':>9}   {'시':>3}{'MAE':>9}{'편의%':>9}")
    for i in range(1, 13):
        r1 = s[s["hour_n"] == i]
        r2 = s[s["hour_n"] == i + 12]
        print(f"  {i:>3}{mae(r1['예측'], r1['실적']):>9.2f}"
              f"{bias(r1['예측'], r1['실적']):>9.2f}   "
              f"{i + 12:>3}{mae(r2['예측'], r2['실적']):>9.2f}"
              f"{bias(r2['예측'], r2['실적']):>9.2f}")

    # ── ④ 월별 주간 편의 추이 ──────────────────
    print("\n■ ④ 월별 주간(10~16시) 편의 추이 — 운영 이력")
    h["ym"] = h["date"].str[:7]
    hd = h[h["hour_n"].isin(DAY_HOURS)]
    print(f"  {'기준월':<9}{'일수':>6}{'MAE':>9}{'편의%':>9}")
    print("  " + "-" * 34)
    for ym, g in hd.groupby("ym"):
        print(f"  {ym:<9}{g['date'].nunique():>6}{mae(g['예측'], g['실적']):>9.2f}"
              f"{bias(g['예측'], g['실적']):>9.2f}")
        rep(f"주간 편의 {ym}", f"MAE {mae(g['예측'], g['실적']):.2f}원",
            f"편의 {bias(g['예측'], g['실적']):+.2f}%")
    print("  " + "-" * 34)


def load_dataset(t):
    d = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False)
    d["date"] = d["date"].astype(str)
    return d


# ── 기온계 변수 A/B ───────────────────────────
def run_ab(t, start, cycle):
    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("[중단] lightgbm 미설치")

    src = os.path.join(PROCESSED, f"smp_features_{t}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}.csv")
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    dd = pd.read_csv(dic, encoding="utf-8-sig")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    base = [c for c in df.columns if c in ok and c not in EXCLUDE]
    base = [c for c in base if c in ("daytype", "season")
            or pd.api.types.is_numeric_dtype(df[c])]
    add = [c for c in TEMP_VARS if c in df.columns and df[c].notna().any()]
    df = df.dropna(subset=[t]).sort_values(["date", "hour_n"]).reset_index(drop=True)
    ho = df[df["date"] >= start]
    days = sorted(ho["date"].unique())

    print("\n■ ⑤ 기온계 변수 A/B — 운영 동일 조건")
    rep("기준 변수", f"{len(base)}개")
    rep("추가 변수", f"{len(add)}개", " · ".join(add) or "없음")
    rep("홀드아웃", f"{days[0]} ~ {days[-1]}", f"{len(days)}일 / {len(ho):,}행")
    rep("재학습 주기", f"{cycle}일", "운영 조건")
    if not add:
        print("  [중단] 추가 대상 변수 없음")
        return

    def make_xy(d, feats):
        X = d[feats].copy()
        for c in feats:
            if c in ("month", "dow", "hour_n", "daytype", "season") \
               or X[c].dtype == object:
                X[c] = X[c].astype("category")
        return X, d[t]

    print(f"\n  {'구성':<10}{'재학습':>7}{'전체MAE':>9}{'9월MAE':>9}"
          f"{'주간MAE':>9}{'주간편의%':>10}")
    print("  " + "-" * 56)
    res = {}
    for nm, feats in (("기준", base), ("기온투입", base + add)):
        preds, acts, dts, hrs, n_fit = [], [], [], [], 0
        model, last = None, None
        for d0 in days:
            gap = None if last is None else \
                (pd.to_datetime(d0) - pd.to_datetime(last)).days
            if last is None or gap >= cycle:
                tr = df[(df["date"] >= TRAIN_START) & (df["date"] < d0)]
                if len(tr) < 5000:
                    continue
                X, y = make_xy(tr, feats)
                model = lgb.LGBMRegressor(
                    n_estimators=600, learning_rate=0.05, num_leaves=63,
                    min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
                    random_state=SEED, verbose=-1).fit(X, y)
                n_fit += 1
                last = d0
            if model is None:
                continue
            seg = ho[ho["date"] == d0]
            X, _ = make_xy(seg, feats)
            preds.append(model.predict(X))
            acts.append(seg[t].values)
            dts.append(np.repeat(d0, len(seg)))
            hrs.append(seg["hour_n"].values)
        p = np.concatenate(preds); a = np.concatenate(acts)
        dt = np.concatenate(dts); hr = np.concatenate(hrs)
        sep = np.char.startswith(dt.astype(str), "2026-09")
        dayi = np.isin(hr, DAY_HOURS)
        row = {"전체MAE": mae(p, a),
               "9월MAE": mae(p[sep], a[sep]) if sep.any() else np.nan,
               "주간MAE": mae(p[dayi], a[dayi]),
               "주간편의": bias(p[dayi], a[dayi])}
        res[nm] = row
        print(f"  {nm:<10}{n_fit:>7}{row['전체MAE']:>9.2f}{row['9월MAE']:>9.2f}"
              f"{row['주간MAE']:>9.2f}{row['주간편의']:>10.2f}")
        rep(f"구성 {nm}", f"MAE {row['전체MAE']:.2f}",
            f"9월 {row['9월MAE']:.2f} / 주간 {row['주간MAE']:.2f} / "
            f"주간편의 {row['주간편의']:+.2f}%")
    print("  " + "-" * 56)

    if len(res) == 2:
        b, c = res["기준"], res["기온투입"]
        print("\n■ ⑥ 판정 — 기준 대비")
        for k in ("전체MAE", "9월MAE", "주간MAE"):
            print(f"    {k:<8} {b[k]:.2f} → {c[k]:.2f} ({c[k] - b[k]:+.2f}원)")
            rep(f"개선 {k}", f"{c[k] - b[k]:+.2f}원", f"{b[k]:.2f} → {c[k]:.2f}")
        print(f"    {'주간편의':<8} {b['주간편의']:+.2f}% → {c['주간편의']:+.2f}%")
        print("\n    ※ 채택 기준 : 9월·주간 MAE 개선 + 전체 MAE 비악화")


def main():
    ap = argparse.ArgumentParser(description="계절 전환기 과대예측 검증")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--date", default=None, help="점검 대상일 YYYY-MM-DD")
    ap.add_argument("--ab", action="store_true", help="기온계 변수 A/B 실증")
    ap.add_argument("--start", default="2026-06-01", help="A/B 홀드아웃 시작일")
    ap.add_argument("--cycle", type=int, default=7, help="재학습 주기(일)")
    args = ap.parse_args()
    t = args.target
    td = args.date or (datetime.now().strftime("%Y-%m-%d"))

    print("=" * 78)
    print(" step8_season_test — 계절 전환기 과대예측 검증 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    diagnose(t, td)
    if args.ab:
        run_ab(t, args.start, args.cycle)

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_season_test_{t}_{ts}.csv")
    pd.DataFrame(_log).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
