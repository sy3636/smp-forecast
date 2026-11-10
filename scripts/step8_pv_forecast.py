# -*- coding: utf-8 -*-
"""
step8_pv_forecast.py — 신재생(태양광) 출력 전망 모형 (v1.0)
==============================================================================
목적
  · D+1 시간대별 태양광 출력을 자체 전망하여 순부하(수요 − 신재생) 변수를 파생
  · 2026.8.27 사내 시연 지적 ⑨ 및 권영옥 교수 자문(기상 데이터 재검토)에 대응

배경
  · 신재생 **실적**은 기 보유(2021-01~2026-06, 47,688행)하나 예측 시각(14~15시)에는
    익일 출력을 알 수 없어 피처사전상 11종 전량이 운영 불가(N)로 분류되어 있음
  · 사내·전력거래소 모두 **전망치 미보유** → 자체 전망 모형 구축이 유일한 경로
  · 기상 변수는 SMP 직접 투입 시 기각되었으나(2026.8.6), 신재생 출력을 매개로 하는
    경로는 미검증 상태였음

설계 — 2개 시나리오 병행
  · 시나리오 A(상한) : 일사·일조 **실측** 포함 — 달성 가능한 이론적 최대 성능
  · 시나리오 B(운영) : 기상청 단기예보 가용 변수만 사용
        · 단기예보(VilageFcst)는 일사·일조를 제공하지 않음
        · 가용 : 기온(TMP)·습도(REH)·풍속(WSD)·강수(PCP)  ※ 하늘상태는 미보유
  · 두 성능의 차이가 곧 「일사 예보 부재로 잃는 성능」이며, 향후 일사 예보
    확보(기상청 신규 API·위성자료 등) 타당성 판단의 근거가 됨

변수
  · 기상  : temp_land, humid_land, wind_spd_land, rain_land (+ A안 : solar_land, sunshine_land)
  · 달력  : 월·시간·요일 및 순환 인코딩(sin/cos)
  · 설비  : cap_proxy — 최근 30일 시간대별 최대 출력(설비용량 증가 추세 대리)
            · 실제 설비용량 시계열 미보유에 따른 대리지표
            · 예측 시각에 확보 가능(과거 실적 기반)

산출
  · data/processed/pv_forecast_eval.csv — 시나리오별 성능
  · ml/models/pv_land_{scenario}.pkl     — 학습 모델
  · logs/step8_pv_forecast_{일시}.csv

실행
  python scripts\\step8_pv_forecast.py
  python scripts\\step8_pv_forecast.py --holdout 2026-01-01

버전 : v1.0 (2026-08-31)
"""
import os
import sys
import argparse
import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("[중단] lightgbm 미설치 — pip install lightgbm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERIM = os.path.join(BASE_DIR, "data", "interim")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

W_SRC = os.path.join(INTERIM, "weather_hourly.csv")
P_SRC = os.path.join(INTERIM, "api_pv_hourly.csv")
OUT = os.path.join(PROCESSED, "pv_forecast_eval.csv")

# 시나리오별 기상 변수 — B안은 기상청 단기예보 가용 항목만
WX_OPERATIONAL = ["temp_land", "humid_land", "wind_spd_land", "rain_land"]
WX_ACTUAL_ONLY = ["solar_land", "sunshine_land"]

SEED = 42
_rows_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<30} {값:<26} {비고}")
    _rows_log.append({"항목": 항목, "값": 값, "비고": 비고})


def build(holdout):
    """학습 데이터 구성 — 기상 × 태양광 실적 결합 및 파생변수 생성"""
    w = pd.read_csv(W_SRC, encoding="utf-8-sig")
    p = pd.read_csv(P_SRC, encoding="utf-8-sig")
    d = w.merge(p[["date", "hour_n", "pv_land"]], on=["date", "hour_n"], how="inner")
    d = d.dropna(subset=["pv_land"]).reset_index(drop=True)

    dt = pd.to_datetime(d["date"])
    d["month"] = dt.dt.month
    d["dow"] = dt.dt.dayofweek
    d["doy"] = dt.dt.dayofyear
    # 순환 인코딩 — 시간·계절의 주기성 반영
    d["hour_sin"] = np.sin(2 * np.pi * d["hour_n"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour_n"] / 24)
    d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365)

    # 설비용량 대리지표 — 최근 30일 동시간대 최대 출력
    #  · 실제 설비용량 시계열 미보유. 예측 시각에 확보 가능한 과거 실적으로 대체
    d = d.sort_values(["hour_n", "date"]).reset_index(drop=True)
    d["cap_proxy"] = (d.groupby("hour_n")["pv_land"]
                        .transform(lambda s: s.shift(1).rolling(30, min_periods=5).max()))
    d = d.sort_values(["date", "hour_n"]).reset_index(drop=True)
    d = d.dropna(subset=["cap_proxy"]).reset_index(drop=True)

    d["is_holdout"] = d["date"] >= holdout
    return d


def run(d, feats, tag):
    """단일 시나리오 학습·평가"""
    tr = d[~d["is_holdout"]]
    te = d[d["is_holdout"]]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
                          random_state=SEED, verbose=-1)
    m.fit(tr[feats], tr["pv_land"])
    pred = np.clip(m.predict(te[feats]), 0, None)
    act = te["pv_land"].values

    # 야간(출력 0 근방) 제외 후 평가 — 태양광은 주간에만 유효
    day = act > 50
    mae = np.abs(pred - act).mean()
    rmse = float(np.sqrt(((pred - act) ** 2).mean()))
    mape = float((np.abs(pred[day] - act[day]) / act[day]).mean() * 100)
    nmae = mae / act.max() * 100          # 최대출력 대비 정규화 오차
    return m, {"시나리오": tag, "학습행": len(tr), "검증행": len(te),
               "MAE(MW)": round(mae, 1), "RMSE(MW)": round(rmse, 1),
               "주간MAPE(%)": round(mape, 2), "nMAE(%)": round(nmae, 2)}, pred, act


def save_log(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(LOG_DIR, f"{name}_{ts}.csv")
    pd.DataFrame(_rows_log).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def apply_operational(fb):
    """저장 B안 모델로 weather 전 구간 전망."""
    import pickle
    mp = os.path.join(MODEL_DIR, "pv_land_B_운영.pkl")
    if not os.path.exists(mp):
        sys.exit("[중단] B안 모델 없음 — python scripts\\step8_pv_forecast.py 선행")
    with open(mp, "rb") as fh:
        mb = pickle.load(fh)
    w = pd.read_csv(W_SRC, encoding="utf-8-sig")
    dt = pd.to_datetime(w["date"])
    w["month"] = dt.dt.month
    w["dow"] = dt.dt.dayofweek
    w["doy"] = dt.dt.dayofyear
    w["hour_sin"] = np.sin(2 * np.pi * w["hour_n"] / 24)
    w["hour_cos"] = np.cos(2 * np.pi * w["hour_n"] / 24)
    w["doy_sin"] = np.sin(2 * np.pi * w["doy"] / 365)
    w["doy_cos"] = np.cos(2 * np.pi * w["doy"] / 365)
    if os.path.exists(P_SRC):
        pva = pd.read_csv(P_SRC, encoding="utf-8-sig")
        cap = (pva.sort_values(["hour_n", "date"])
               .groupby("hour_n")["pv_land"]
               .transform(lambda s: s.shift(1).rolling(30, min_periods=5).max()))
        pva = pva.assign(cap_proxy=cap)
        last_cap = pva.dropna(subset=["cap_proxy"]).groupby("hour_n")["cap_proxy"].last()
        w["cap_proxy"] = w["hour_n"].map(last_cap)
    else:
        w["cap_proxy"] = np.nan
    use = [c for c in fb if c in w.columns]
    pr = np.clip(mb.predict(w[use].fillna(0)), 0, None)
    f = pd.DataFrame({"date": w["date"], "hour_n": w["hour_n"], "pv_fcst": pr})
    fp = os.path.join(INTERIM, "pv_forecast_hourly.csv")
    f.to_csv(fp, index=False, encoding="utf-8-sig")
    return fp, f


def main():
    ap = argparse.ArgumentParser(description="신재생(태양광) 출력 전망 모형")
    ap.add_argument("--holdout", default="2026-01-01", help="검증 시작일")
    ap.add_argument("--no-save", action="store_true", help="모델 미저장")
    ap.add_argument("--emit", action="store_true",
                    help="전 구간 전망값 산출 — 시계열 교차예측 방식(누출 방지). "
                         "산출 : data/interim/pv_forecast_hourly.csv")
    ap.add_argument("--apply", action="store_true",
                    help="저장 모델(B안)로 weather_hourly 전 구간 태양광 전망 산출")
    args = ap.parse_args()

    print("=" * 72)
    print(" step8_pv_forecast — 태양광 출력 전망 모형 v1.0")
    print("=" * 72)

    if not os.path.exists(W_SRC):
        sys.exit(f"[중단] 입력파일 없음 → {W_SRC}")
    if not args.apply and not os.path.exists(P_SRC):
        sys.exit(f"[중단] 입력파일 없음 → {P_SRC}")

    CAL = ["month", "hour_n", "dow", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    fa = WX_OPERATIONAL + WX_ACTUAL_ONLY + CAL + ["cap_proxy"]
    fb = WX_OPERATIONAL + CAL + ["cap_proxy"]

    if args.apply and not args.emit:
        print("\n■ 운영 전망 적용")
        fp, f = apply_operational(fb)
        rep("운영 전망", os.path.relpath(fp, BASE_DIR),
            f"{f['date'].min()} ~ {f['date'].max()} / {len(f):,}행")
        p = save_log("step8_pv_forecast")
        print(f"\n[완료] 로그 → {os.path.relpath(p, BASE_DIR)}")
        return

    print("\n■ ① 데이터 구성")
    d = build(args.holdout)
    rep("결합 데이터", f"{len(d):,}행", f"{d['date'].min()} ~ {d['date'].max()}")
    rep("홀드아웃 기준", args.holdout,
        f"학습 {(~d['is_holdout']).sum():,} / 검증 {d['is_holdout'].sum():,}")

    print("\n■ ② 시나리오별 학습")
    rep("A(상한) 변수", f"{len(fa)}종", "일사·일조 실측 포함")
    rep("B(운영) 변수", f"{len(fb)}종", "기상청 단기예보 가용 항목만")

    res = []
    models = {}
    for tag, feats in [("A_상한", fa), ("B_운영", fb)]:
        m, r, pred, act = run(d, feats, tag)
        res.append(r)
        models[tag] = m

    e = pd.DataFrame(res)
    print("\n■ ③ 성능 비교")
    print(e.to_string(index=False))

    a = e[e["시나리오"] == "A_상한"].iloc[0]
    b = e[e["시나리오"] == "B_운영"].iloc[0]
    print()
    rep("일사 예보 부재 손실", f"nMAE {b['nMAE(%)'] - a['nMAE(%)']:+.2f}%p",
        f"A {a['nMAE(%)']}% → B {b['nMAE(%)']}%")
    rep("운영 성능(B안)", f"nMAE {b['nMAE(%)']}%",
        f"MAE {b['MAE(MW)']}MW / 최대출력 대비")

    print("\n■ ④ 변수 중요도 (B안 상위 8)")
    imp = pd.Series(models["B_운영"].feature_importances_, index=fb).sort_values(ascending=False)
    for k, v in imp.head(8).items():
        print(f"  {k:<18} {v / imp.sum() * 100:5.1f}%")

    print("\n■ ⑤ 산출")
    os.makedirs(PROCESSED, exist_ok=True)
    e.to_csv(OUT, index=False, encoding="utf-8-sig")
    rep("성능 파일", os.path.relpath(OUT, BASE_DIR))
    if not args.no_save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        import pickle
        for tag, m in models.items():
            p = os.path.join(MODEL_DIR, f"pv_land_{tag}.pkl")
            with open(p, "wb") as fh:
                pickle.dump(m, fh)
        rep("모델 저장", f"{len(models)}종", "ml/models/pv_land_*.pkl")

    if args.emit:
        print("\n■ ⑥ 전 구간 전망값 산출 — 시계열 교차예측")
        # 누출 방지 : 각 연도를 예측할 때 그 이전 구간만으로 학습
        d2 = d.copy()
        d2["yr"] = d2["date"].str[:4].astype(int)
        yrs = sorted(d2["yr"].unique())
        out = []
        for y in yrs[1:]:                       # 첫 해는 학습 전용
            tr = d2[d2["yr"] < y]
            te = d2[d2["yr"] == y]
            mm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=30, subsample=0.9,
                                   colsample_bytree=0.9, random_state=SEED, verbose=-1)
            mm.fit(tr[fb], tr["pv_land"])
            pr = np.clip(mm.predict(te[fb]), 0, None)
            out.append(pd.DataFrame({"date": te["date"].values,
                                     "hour_n": te["hour_n"].values,
                                     "pv_fcst": pr}))
            e_ = np.abs(pr - te["pv_land"].values).mean()
            rep(f"  {y}년 예측", f"{len(te):,}행", f"MAE {e_:.1f}MW (학습 {len(tr):,}행)")
        f = pd.concat(out, ignore_index=True)
        fp = os.path.join(INTERIM, "pv_forecast_hourly.csv")
        f.to_csv(fp, index=False, encoding="utf-8-sig")
        rep("전망값 파일", os.path.relpath(fp, BASE_DIR),
            f"{f['date'].min()} ~ {f['date'].max()} / {len(f):,}행")

    if args.apply:
        print("\n■ ⑦ 운영 전망 적용 — weather 전 구간")
        fp, f = apply_operational(fb)
        rep("운영 전망", os.path.relpath(fp, BASE_DIR),
            f"{f['date'].min()} ~ {f['date'].max()} / {len(f):,}행")

    p = save_log("step8_pv_forecast")
    print(f"\n[완료] 로그 → {os.path.relpath(p, BASE_DIR)}")


if __name__ == "__main__":
    main()
