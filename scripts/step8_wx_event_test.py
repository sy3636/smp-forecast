# -*- coding: utf-8 -*-
"""
step8_wx_event_test.py — 강수·저일사 이례일 기상 반영 실증 (v1.0)
==============================================================================
목적
  · 종전 4회 기각은 **전 시간 평균 MAPE** 기준이었음
  · 2026.9.1 전국 강수일에 SMP 실적 190원대 vs 예측 170원대 괴리가 발생
  · 「평균은 안 좋아져도, 재생에너지 불리 이례일의 체계적 저예측은
     기상 변수로 완화되는가」를 분리 검증

설계
  · 기준          : 운영가용 변수만 (기상 미투입) — 현행
  · 기상4종       : 단기예보 가용 항목 실측 대용(기온·습도·풍속·강수) — 상한
  · 이례지시      : 강수일·주간강수합·일사비율·일사부족 — 사건 집중
  · 기상4종+이례  : 병행
  · 이례편향보정  : 기준 예측 + 학습구간 이례일 평균잔차 (해석 용이)

이례일 정의
  · 강수일     : 주간(9~17시) rain_land 합 ≥ 1.0 mm
  · 저일사일   : 주간 일사합 / 학습구간 동월 중앙값 < 0.70
  · 재생불리일 : 강수일 또는 저일사일  (당일 24시간 모두 이례로 표기)
  · 9.1 사례는 저녁 피크까지 오차가 지속되어 **주간 한정 마스킹을 쓰지 않음**

판정
  · 주판정(G-1 준용) : 전체 MAPE 0.10%p 이상 개선, 홀드아웃 2구간 부호 일관
  · 보완전정         : 이례일 MAE 1.0원 이상 개선 AND 비이례일 악화 < 0.05%p
  · 기상 실측은 예측 시점 미확보 → 결과는 **이론적 상한**. 채택 시 단기예보 이관 필요

산출
  · data/processed/wx_event_test.csv
  · data/processed/wx_event_hourly.csv   (홀드아웃 잔차, 이례 표기)
  · logs/step8_wx_event_test_{일시}.csv

실행
  python scripts\\step8_wx_event_test.py

버전 : v1.0 (2026-09-01)
"""
import os
import sys
import gc
import json
import glob
import argparse
import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("[중단] lightgbm 미설치 — pip install lightgbm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
RAW_ASOS = os.path.join(BASE_DIR, "data", "raw", "asos")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")

SRC = os.path.join(PROCESSED, "smp_features_smp_land.csv")
DICT = os.path.join(PROCESSED, "feature_dictionary_smp_land.csv")
WSRC = os.path.join(INTERIM, "weather_hourly.csv")
OUT = os.path.join(PROCESSED, "wx_event_test.csv")
OUT_H = os.path.join(PROCESSED, "wx_event_hourly.csv")

TARGET = "smp_land"
DAY_LO, DAY_HI = 9, 17
RAIN_MM = 1.0
SOLAR_RATIO = 0.70
WX4 = ["temp_land", "humid_land", "wind_spd_land", "rain_land"]
EVT = ["is_rain_day", "rain_daytime", "solar_ratio", "solar_deficit"]
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju", "year",
           "unit_id", "unit_name", "grp", "pv_source", "holiday_name"}

STATIONS = {
    "108": 0.30, "112": 0.10, "133": 0.13, "143": 0.13,
    "156": 0.10, "159": 0.17, "105": 0.07,
}
LAND_IDS = list(STATIONS)
FIELDS = {"ta": "temp", "hm": "humid", "ws": "wind_spd",
          "icsr": "solar", "ss": "sunshine", "rn": "rain"}

HOLDOUTS = [("2026-01~08", "2026-01-01", "2026-09-01"),
            ("2025-09~12", "2025-09-01", "2026-01-01")]

_rows_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<28} {값:<28} {비고}", flush=True)
    _rows_log.append({"항목": 항목, "값": 값, "비고": 비고})


def mape(pred, act):
    act = np.asarray(act, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = act > 1
    return float((np.abs(pred[m] - act[m]) / act[m]).mean() * 100)


def mae(pred, act):
    act = np.asarray(act, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = act > 1
    return float(np.abs(pred[m] - act[m]).mean())


def append_asos_month(weather, ym="202608"):
    """미파싱 월 ASOS JSON을 벡터 집약하여 weather_hourly에 추가."""
    files = sorted(glob.glob(os.path.join(RAW_ASOS, f"*_{ym}.json")))
    if not files:
        return weather, 0
    recs = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        items = (((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
        if isinstance(items, dict):
            items = [items]
        recs.extend(items)
    if not recs:
        return weather, 0
    df = pd.DataFrame(recs)
    df["stnId"] = df["stnId"].astype(str).str.strip()
    df["_ts"] = pd.to_datetime(df["tm"], errors="coerce")
    df = df.dropna(subset=["_ts"])
    for src, dst in FIELDS.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    for v in ("rain", "solar", "sunshine"):
        df[v] = df[v].fillna(0.0)
    df["date"] = df["_ts"].dt.strftime("%Y-%m-%d")
    df["hour_n"] = df["_ts"].dt.hour + 1
    df["_w"] = df["stnId"].map(STATIONS).fillna(0.0)
    land = df[df["stnId"].isin(LAND_IDS)].copy()
    jeju = df[df["stnId"] == "184"].copy()

    def wavg(g, col):
        m = g[col].notna()
        w = g.loc[m, "_w"]
        if w.sum() == 0:
            return np.nan
        return float((g.loc[m, col] * w).sum() / w.sum())

    out = None
    for col in FIELDS.values():
        a = (land.groupby(["date", "hour_n"], sort=False)
             .apply(lambda g, c=col: wavg(g, c), include_groups=False)
             .rename(f"{col}_land").reset_index())
        b = (jeju.groupby(["date", "hour_n"], sort=False)[col].mean()
             .rename(f"{col}_jeju").reset_index())
        m = a.merge(b, on=["date", "hour_n"], how="outer")
        out = m if out is None else out.merge(m, on=["date", "hour_n"], how="outer")
    drop = [c for c in ("wind_spd_jeju", "sunshine_jeju", "rain_jeju") if c in out.columns]
    out = out.drop(columns=drop)
    weather = pd.concat([weather, out], ignore_index=True)
    weather = weather.drop_duplicates(subset=["date", "hour_n"], keep="last")
    weather = weather.sort_values(["date", "hour_n"]).reset_index(drop=True)
    return weather, len(out)


def fit_predict(tr, te, feats, seed):
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                          random_state=seed, verbose=-1, n_jobs=2, force_col_wise=True)
    m.fit(tr[feats], tr[TARGET])
    p_tr = m.predict(tr[feats])
    p_te = m.predict(te[feats])
    del m
    gc.collect()
    return p_tr, p_te


def scores(pred, act, event, day):
    """전체·이례·비이례·주간 이례 MAPE/MAE."""
    pred, act, event, day = map(np.asarray, (pred, act, event, day))
    out = {
        "전체MAPE": mape(pred, act),
        "전체MAE": mae(pred, act),
        "이례MAPE": mape(pred[event], act[event]) if event.any() else np.nan,
        "이례MAE": mae(pred[event], act[event]) if event.any() else np.nan,
        "비이례MAPE": mape(pred[~event], act[~event]) if (~event).any() else np.nan,
        "비이례MAE": mae(pred[~event], act[~event]) if (~event).any() else np.nan,
    }
    ed = event & day
    if ed.any():
        out["이례주간MAPE"] = mape(pred[ed], act[ed])
        out["이례주간MAE"] = mae(pred[ed], act[ed])
        out["이례주간잔차"] = float((act[ed] - pred[ed]).mean())
    else:
        out["이례주간MAPE"] = np.nan
        out["이례주간MAE"] = np.nan
        out["이례주간잔차"] = np.nan
    out["이례시간"] = int(event.sum())
    out["이례잔차"] = float((act[event] - pred[event]).mean()) if event.any() else np.nan
    return out


def case_0901():
    """운영 산출물 기준 2026-09-01 오차 — 모델 재학습과 무관."""
    p = os.path.join(PRED_DIR, "SMP예측_20260901.csv")
    w = os.path.join(PRED_DIR, "SMP주간예측_20260831.csv")
    if not os.path.exists(p):
        return None
    h = pd.read_csv(p, encoding="utf-8-sig")
    h["hour_n"] = h["hour_n"].astype(int)
    act, pr = h["실적_smp_land"].astype(float), h["예측_smp_land"].astype(float)
    rows = []
    for lo, hi, lab in ((1, 24, "전일"), (9, 17, "주간9-17"), (9, 21, "주간+저녁9-21")):
        m = h["hour_n"].between(lo, hi)
        rows.append({
            "구분": f"시간별예측/{lab}",
            "예측평균": round(float(pr[m].mean()), 2),
            "실적평균": round(float(act[m].mean()), 2),
            "MAE": round(mae(pr[m], act[m]), 2),
            "MAPE": round(mape(pr[m], act[m]), 2),
            "잔차": round(float((act[m] - pr[m]).mean()), 2),
            "실적최대": round(float(act[m].max()), 2),
        })
    if os.path.exists(w):
        wk = pd.read_csv(w, encoding="utf-8-sig")
        d1 = wk[(wk["대상일"] == "2026-09-01") & (wk["지평"] == "D+1")].copy()
        if len(d1):
            d1["거래시간"] = d1["거래시간"].astype(int)
            d1 = d1.merge(h[["hour_n", "실적_smp_land"]],
                          left_on="거래시간", right_on="hour_n", how="left")
            prw = d1["예측"].astype(float)
            acw = d1["실적_smp_land"].astype(float)
            for lo, hi, lab in ((1, 24, "전일"), (9, 21, "주간+저녁9-21")):
                m = d1["거래시간"].between(lo, hi)
                rows.append({
                    "구분": f"주간D+1/{lab}",
                    "예측평균": round(float(prw[m].mean()), 2),
                    "실적평균": round(float(acw[m].mean()), 2),
                    "MAE": round(mae(prw[m], acw[m]), 2),
                    "MAPE": round(mape(prw[m], acw[m]), 2),
                    "잔차": round(float((acw[m] - prw[m]).mean()), 2),
                    "실적최대": round(float(acw[m].max()), 2),
                })
    hourly = h[["hour_n"]].copy()
    hourly["예측"] = pr
    hourly["실적"] = act
    hourly["잔차"] = act - pr
    return rows, hourly


def load():
    print("\n■ ① 데이터 적재")
    w = pd.read_csv(WSRC, encoding="utf-8-sig")
    n_add = 0
    if str(w["date"].max()) < "2026-08-31":
        w, n_add = append_asos_month(w, "202608")
        tmp = WSRC + ".tmp"
        w.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, WSRC)
    rep("기상", f"{len(w):,}행", f"{w['date'].min()} ~ {w['date'].max()}"
        + (f" / 8월 추가 {n_add}행" if n_add else ""))

    dic = pd.read_csv(DICT, encoding="utf-8-sig")
    key = "변수명" if "변수명" in dic.columns else dic.columns[0]
    feats = [c for c in dic[dic["운영가용"] == "Y"][key]
             if c != TARGET and c not in EXCLUDE]
    head = pd.read_csv(SRC, encoding="utf-8-sig", nrows=0)
    wx_keep = [c for c in (WX4 + ["solar_land", "sunshine_land"]) if c in w.columns]
    use = [c for c in set(feats + [TARGET, "date", "hour_n"]) if c in head.columns]
    d = pd.read_csv(SRC, encoding="utf-8-sig", usecols=use, low_memory=False)
    d = d[d[TARGET] > 1].reset_index(drop=True)
    drop_wx = [c for c in wx_keep if c in d.columns]
    if drop_wx:
        d = d.drop(columns=drop_wx)
    d = d.merge(w[["date", "hour_n"] + wx_keep], on=["date", "hour_n"], how="left")
    feats = [c for c in feats if c in d.columns]
    has_wx = d["rain_land"].notna() if "rain_land" in d.columns else pd.Series(False, index=d.index)
    rep("피처", f"{len(d):,}행 / 운영가용 {len(feats)}종",
        f"{d['date'].min()} ~ {d['date'].max()}")
    rep("기상 결합", f"{has_wx.mean() * 100:.1f}%",
        f"결측 {int((~has_wx).sum()):,}행 — 평가에서 제외")

    dt = d["date"].astype(str)
    d["month_n"] = pd.to_datetime(dt).dt.month.astype("int8")
    day = d["hour_n"].between(DAY_LO, DAY_HI)
    g = d.loc[day].groupby("date", sort=False)
    rain_dt = g["rain_land"].sum().rename("rain_daytime")
    solar_dt = g["solar_land"].sum().rename("solar_daytime")
    d = d.merge(rain_dt, left_on="date", right_index=True, how="left")
    d = d.merge(solar_dt, left_on="date", right_index=True, how="left")
    d["is_rain_day"] = (d["rain_daytime"] >= RAIN_MM).astype("float32")
    d["solar_ratio"] = np.nan
    d["solar_deficit"] = np.nan

    for c in ("daytype", "season", "month"):
        if c in d.columns:
            d[c] = d[c].astype("category")
    for c in feats + WX4 + EVT + ["solar_land"]:
        if c not in d.columns:
            continue
        if d[c].dtype == object or str(d[c].dtype) in ("string", "str"):
            d[c] = d[c].astype("category")
        elif d[c].dtype == "float64":
            d[c] = d[c].astype("float32")
    return d, dt, feats, has_wx.values, day.values


def add_solar_ratio(d, m_tr):
    """학습구간 동월 주간일사 중앙값 대비 비율 — 누설 방지."""
    tr = d[m_tr]
    med = (tr.drop_duplicates("date")
           .groupby("month_n")["solar_daytime"].median())
    ratio = d["solar_daytime"] / d["month_n"].map(med)
    d = d.copy()
    d["solar_ratio"] = ratio.astype("float32")
    d["solar_deficit"] = (1.0 - d["solar_ratio"]).clip(lower=0).astype("float32")
    d["is_low_solar"] = (d["solar_ratio"] < SOLAR_RATIO).astype("float32")
    d["is_event"] = ((d["is_rain_day"] >= 0.5) | (d["is_low_solar"] >= 0.5)).astype(bool)
    d.loc[d["rain_daytime"].isna(), "is_event"] = False
    return d


def run_holdout(d, dt, feats, has_wx, day, lab, tr_end, te_end, seed):
    m_tr = (dt < tr_end).values & has_wx
    m_te = (dt >= tr_end).values & (dt < te_end).values & has_wx
    d = add_solar_ratio(d, m_tr)
    tr, te = d[m_tr], d[m_te]
    event_te = te["is_event"].values
    day_te = day[m_te]
    act = te[TARGET].values
    n_event_days = int(te.loc[te["is_event"]].drop_duplicates("date").shape[0])
    n_days = int(te.drop_duplicates("date").shape[0])

    print(f"\n  [{lab}] 학습 {int(m_tr.sum()):,} / 검증 {int(m_te.sum()):,} "
          f"/ 이례일 {n_event_days}/{n_days}일 "
          f"({n_event_days / max(n_days, 1) * 100:.1f}%)", flush=True)

    variants = []
    # 기준
    p_tr, p_te = fit_predict(tr, te, feats, seed)
    variants.append(("기준", p_te, scores(p_te, act, event_te, day_te)))

    # 이례편향보정 — 학습 이례일 평균잔차
    ev_tr = tr["is_event"].values
    bias = float((tr[TARGET].values[ev_tr] - p_tr[ev_tr]).mean()) if ev_tr.any() else 0.0
    p_adj = p_te + np.where(event_te, bias, 0.0)
    variants.append(("이례편향보정", p_adj, scores(p_adj, act, event_te, day_te)))
    variants[-1][2]["보정량"] = round(bias, 3)

    for tag, extra in [("기상4종", WX4),
                       ("이례지시", EVT),
                       ("기상4종+이례", WX4 + EVT)]:
        extra = [c for c in extra if c in tr.columns]
        _, p = fit_predict(tr, te, feats + extra, seed)
        variants.append((tag, p, scores(p, act, event_te, day_te)))

    rows = []
    hourly = te[["date", "hour_n", TARGET, "is_event", "is_rain_day",
                 "rain_daytime", "solar_ratio"]].copy()
    hourly["홀드아웃"] = lab
    hourly["잔차_기준"] = act - variants[0][1]
    for tag, p, sc in variants:
        rec = {"홀드아웃": lab, "조건": tag, **{k: (round(v, 3) if isinstance(v, float) else v)
                                              for k, v in sc.items()}}
        rec["이례일수"] = n_event_days
        rec["검증일수"] = n_days
        rows.append(rec)
        print(f"    {tag:<12} 전체 {sc['전체MAPE']:.2f}% / "
              f"이례MAE {sc['이례MAE']:.2f}원 / "
              f"이례잔차 {sc['이례잔차']:+.2f}원 / "
              f"비이례 {sc['비이례MAPE']:.2f}%", flush=True)
    # 기준 잔차 진단
    r = hourly["잔차_기준"].values
    ev = hourly["is_event"].astype(bool).values
    rain = hourly["is_rain_day"].fillna(0).values >= 0.5
    if ev.any():
        rep(f"[{lab}] 이례 잔차(기준)", f"{r[ev].mean():+.2f}원",
            f"비이례 {r[~ev].mean():+.2f}원 / 강수일 {r[rain].mean():+.2f}원" if rain.any()
            else f"비이례 {r[~ev].mean():+.2f}원")
    return rows, hourly, variants


def save_log(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(LOG_DIR, f"{name}_{ts}.csv")
    pd.DataFrame(_rows_log).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def judge(e):
    print("\n■ ③ 판정")
    base = e[e["조건"] == "기준"]
    for lab, _, _ in HOLDOUTS:
        b = base[base["홀드아웃"] == lab].iloc[0]
        print(f"\n  [{lab}] 기준 전체 {b['전체MAPE']:.2f}% / 이례MAE {b['이례MAE']:.2f}원")
        for tag in ("이례편향보정", "기상4종", "이례지시", "기상4종+이례"):
            g = e[(e["홀드아웃"] == lab) & (e["조건"] == tag)]
            if g.empty:
                continue
            r = g.iloc[0]
            d_all = r["전체MAPE"] - b["전체MAPE"]
            d_evt = r["이례MAE"] - b["이례MAE"]
            d_ne = r["비이례MAPE"] - b["비이례MAPE"]
            g1 = d_all <= -0.10
            aux = (d_evt <= -1.0) and (d_ne < 0.05)
            if g1:
                verd = "채택(G-1)"
            elif aux:
                verd = "보완채택(이례일)"
            else:
                verd = "기준 미달"
            rep(f"[{lab}] {tag}",
                f"전체 {d_all:+.2f}%p / 이례MAE {d_evt:+.2f}원",
                f"비이례 {d_ne:+.2f}%p — {verd}")
    # 양 구간 일관
    print("\n  · 양 구간 일관 여부")
    for tag in ("이례편향보정", "기상4종", "이례지시", "기상4종+이례"):
        ds = []
        for lab, _, _ in HOLDOUTS:
            b = base[base["홀드아웃"] == lab].iloc[0]
            r = e[(e["홀드아웃"] == lab) & (e["조건"] == tag)].iloc[0]
            ds.append(r["전체MAPE"] - b["전체MAPE"])
        ok = all(x < 0 for x in ds)
        rep(f"일관 {tag}", f"{ds[0]:+.2f} / {ds[1]:+.2f}%p",
            "부호 일관" if ok else "부호 역전")


def main():
    ap = argparse.ArgumentParser(description="강수·저일사 이례일 기상 반영 실증")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 74)
    print(" step8_wx_event_test — 강수·저일사 이례일 기상 반영 실증 v1.0")
    print("=" * 74)

    for f in (SRC, DICT, WSRC):
        if not os.path.exists(f):
            sys.exit(f"[중단] 입력파일 없음 → {f}")

    print("\n■ ⓪ 2026-09-01 운영 예측 사례")
    c = case_0901()
    if c:
        rows0901, hourly0901 = c
        for r in rows0901:
            rep(r["구분"], f"실적 {r['실적평균']} / 예측 {r['예측평균']}",
                f"MAE {r['MAE']}원 MAPE {r['MAPE']}% 잔차 {r['잔차']:+}원 최대 {r['실적최대']}")
        os.makedirs(PROCESSED, exist_ok=True)
        hourly0901.to_csv(os.path.join(PROCESSED, "wx_event_0901_hourly.csv"),
                          index=False, encoding="utf-8-sig")
    else:
        rep("0901 사례", "예측 파일 없음", "")

    d, dt, feats, has_wx, day = load()
    wx4_ok = [c for c in WX4 if c in d.columns]
    rep("이례 정의", f"주간강수≥{RAIN_MM}mm 또는 일사비율<{SOLAR_RATIO}",
        f"주간 {DAY_LO}~{DAY_HI}시 / 기상4종 {', '.join(wx4_ok)}")

    print("\n■ ② 홀드아웃 실증")
    all_rows, hours = [], []
    for lab, tr_end, te_end in HOLDOUTS:
        rows, hourly, _ = run_holdout(d, dt, feats, has_wx, day,
                                      lab, tr_end, te_end, args.seed)
        all_rows.extend(rows)
        hours.append(hourly)

    e = pd.DataFrame(all_rows)
    judge(e)

    print("\n■ ④ 산출")
    os.makedirs(PROCESSED, exist_ok=True)
    e.to_csv(OUT, index=False, encoding="utf-8-sig")
    h = pd.concat(hours, ignore_index=True)
    h.to_csv(OUT_H, index=False, encoding="utf-8-sig")
    rep("성능 파일", os.path.relpath(OUT, BASE_DIR), f"{len(e)}개 조합")
    rep("잔차 파일", os.path.relpath(OUT_H, BASE_DIR), f"{len(h):,}행")
    lp = save_log("step8_wx_event_test")
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")


if __name__ == "__main__":
    main()
