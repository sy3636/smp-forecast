# -*- coding: utf-8 -*-
"""
features.py — 시간별 D+1 가격 예측용 변수 표 (1행 = 대상일 1시간)
==============================================================================
누출 방지 — 대상일 i 의 변수는 전날(i−1) 저녁에 확보되는 정보만 사용
  · SMP          : i−1 일 이전 (i−1 일 SMP 는 i−2 일 저녁 공표)
  · 수요예측     : i 일 하루전 수요예측 (i−1 일 저녁 공표)
  · 태양광       : 실적 2일 지연 → 「그 달 이전 월별·시간별 이용률 × 설비용량」 계절 전망
  · 원전 용량·연료비 : 변동비순위표(적용 전월 사전 공표), 국제 연료지수는 2개월 지연
  · 기상 실측(기온·일사 등)은 쓰지 않음 — 대상일 값은 예측 시점에 모름
"""
import warnings

import numpy as np
import pandas as pd

from common import DATA, load_hourly, load_monthly, rd, to_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)       # 전부 NaN 인 날의 nanmean 경고

PV_LAG = 2
NUC_DEFAULT = [("0000-01", 23833.0), ("2023-01", 25189.5), ("2024-06", 26641.3)]
NUC_PATTERN = "한빛|한울|고리|월성|새울"
CAL_COLS = ["is_holiday", "is_prev_day_off", "is_next_day_off", "holiday_run_len", "holiday_position",
            "is_bridge_day", "is_major_holiday", "is_major_holiday_period", "is_major_holiday_eve",
            "is_kpx_myeong_adj", "is_kpx_holiday_eve", "is_kpx_holiday_return"]


def nuclear_by_month(yms):
    """변동비순위표의 원전 호기 cap_max 합 — 뒤 달 값으로 채우고 없으면 마지막 값 (crash_model 과 같은 규칙)"""
    m = rd(DATA / "interim" / "src_merit_public.csv", usecols=["ym", "unit_name", "cap_max"])
    s = m[m["unit_name"].str.contains(NUC_PATTERN, na=False)].groupby("ym")["cap_max"].sum().sort_index()
    out = {}
    for ym in yms:
        later = s[s.index >= ym]
        if len(s) and ym >= s.index[0]:
            out[ym] = later.iloc[0] if len(later) else s.iloc[-1]
        else:
            out[ym] = [c for k, c in NUC_DEFAULT if ym >= k][-1]
    return pd.Series(out)


def seasonal_pv(dates, PV):
    """계절 태양광 전망 (일수×24) — 그 달 1일 이전(공표 지연 반영) 자료로만 만든 월·시간별 이용률 × 설비용량"""
    n = len(dates)
    pv_max = np.nanmax(np.where(np.isnan(PV), -np.inf, PV), axis=1)
    pv_max[~np.isfinite(pv_max)] = np.nan
    cap = np.full(n, np.nan)
    last = np.nan
    for i in range(n):
        w = pv_max[max(0, i - PV_LAG - 59): max(0, i - PV_LAG + 1)]
        w = w[~np.isnan(w)]
        if len(w) >= 20:
            last = w.max()
        cap[i] = last
    cf = np.clip(PV / cap[:, None], 0, 1.2)
    months = np.array([int(d[5:7]) for d in dates]) - 1
    allsum = np.zeros((12, 24)); allcnt = np.zeros((12, 24))
    for i in range(n):
        ok = ~np.isnan(cf[i])
        allsum[months[i], ok] += cf[i, ok]; allcnt[months[i], ok] += 1
    run_s = np.zeros((12, 24)); run_c = np.zeros((12, 24))
    day_num = pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[D]").astype(int)
    month_start = pd.to_datetime(pd.Series(dates)).dt.to_period("M").dt.start_time.to_numpy()
    month_start = month_start.astype("datetime64[D]").astype(int)
    out = np.full((n, 24), np.nan)
    added = 0
    for i in range(n):
        cutoff = month_start[i] - (PV_LAG - 1)
        while added < n and day_num[added] < cutoff:
            ok = ~np.isnan(cf[added])
            run_s[months[added], ok] += cf[added, ok]; run_c[months[added], ok] += 1
            added += 1
        k = months[i]
        src_s, src_c = (run_s[k], run_c[k]) if run_c[k].min() >= 50 else (allsum[k], allcnt[k])
        out[i] = np.where(src_c > 0, src_s / np.maximum(src_c, 1), np.nan) * cap[i]
    return out


def last_same(off, P, mode="row"):
    """대상일과 휴일 여부가 같은 직전 날의 값 (행 인덱스)"""
    n = len(off)
    idx = np.full(n, -1)
    last = {0: -1, 1: -1}
    for i in range(n):
        idx[i] = last[int(off[i])]
        if not np.isnan(P[i]).all():
            last[int(off[i])] = i
    return idx


def build():
    h = load_hourly()
    dates = h["date"].iloc[::24].to_numpy()
    n = len(dates)
    P, Dm, PV = to_matrix(h, "smp"), to_matrix(h, "dem"), to_matrix(h, "pv")
    off = h["off"].to_numpy().reshape(-1, 24)[:, 0]
    yms = np.array([d[:7] for d in dates])
    mon = load_monthly()
    nuc = nuclear_by_month(sorted(set(yms)))
    pvc = seasonal_pv(dates, PV)

    def sh(M, k):
        """k 일 전 행렬 (앞쪽은 NaN)"""
        out = np.full_like(M, np.nan)
        out[k:] = M[:-k]
        return out

    with np.errstate(all="ignore"):
        P1, P2, P7 = sh(P, 1), sh(P, 2), sh(P, 7)
        D1, D7 = sh(Dm, 1), sh(Dm, 7)
        rows = lambda v: np.repeat(np.asarray(v, float), 24)            # 일 단위 → 시간 행
        flat = lambda M: np.asarray(M, float).ravel()
        f = {}
        f["p1"], f["p2"], f["p7"] = flat(P1), flat(P2), flat(P7)
        p1_prev = np.concatenate([P2[:, 23:24], P1[:, :23]], axis=1)
        p1_next = np.concatenate([P1[:, 1:], P1[:, 23:24]], axis=1)
        f["p1_prev"], f["p1_next"] = flat(p1_prev), flat(p1_next)
        f["p1_last"] = rows(P1[:, 23])
        f["p1_mean"], f["p1_max"], f["p1_min"] = rows(np.nanmean(P1, 1)), rows(np.nanmax(P1, 1)), rows(np.nanmin(P1, 1))
        f["p1_std"] = rows(np.nanstd(P1, 1))
        f["p1_mid_min"] = rows(np.nanmin(P1[:, 10:15], 1))
        f["p1_eve_max"] = rows(np.nanmax(P1[:, 16:22], 1))
        f["p2_mean"], f["p7_mean"] = rows(np.nanmean(P2, 1)), rows(np.nanmean(P7, 1))
        ma7 = np.full_like(P, np.nan)
        for i in range(7, n):
            ma7[i] = np.nanmean(P[i - 7:i], axis=0)
        f["pma7"] = flat(ma7)
        f["pma7_mean"] = rows(np.nanmean(ma7, 1))
        dmean = pd.Series(np.nanmean(P, 1))
        f["pma28_mean"] = rows(dmean.shift(1).rolling(28, min_periods=14).mean())
        f["p_ewm_mean"] = rows(dmean.shift(1).ewm(span=7, ignore_na=True).mean())
        same = last_same(off, P)
        PS = np.where((same >= 0)[:, None], P[np.maximum(same, 0)], np.nan)
        f["p_same"] = flat(PS)
        f["p_same_mean"] = rows(np.nanmean(PS, 1))
        f["p_same_mid_min"] = rows(np.nanmin(PS[:, 10:15], 1))
        f["p1_trend"] = f["p1_mean"] - f["p7_mean"]

        f["dem"] = flat(Dm)
        dmean_d = np.nanmean(Dm, 1)
        f["dem_mean"], f["dem_max"], f["dem_min"] = rows(dmean_d), rows(np.nanmax(Dm, 1)), rows(np.nanmin(Dm, 1))
        f["dem_rel"] = f["dem"] / f["dem_mean"]
        f["dem_rank"] = flat(pd.DataFrame(Dm).rank(axis=1).to_numpy())
        f["dem1"] = flat(D1)
        f["dem_ratio1"] = f["dem"] / f["dem1"]
        f["dem_mean_ratio1"] = f["dem_mean"] / rows(np.nanmean(D1, 1))
        f["dem_ratio7"] = f["dem"] / flat(D7)
        dip = np.nanmin(Dm[:, 10:15], 1) / np.nanmean(Dm[:, 7:9], 1)
        f["dem_dip"] = rows(dip)
        f["dem_mid_min"] = rows(np.nanmin(Dm[:, 10:15], 1))
        f["p_per_dem1"] = f["p1"] / f["dem1"] * 1000
        nuc_d = nuc.reindex(yms).to_numpy()
        f["nuc"] = rows(nuc_d)
        f["dem_resid_nuc"] = f["dem"] - f["nuc"]
        f["dem_mid_resid_nuc"] = f["dem_mid_min"] - f["nuc"]
        f["pvc"] = flat(pvc)
        f["net_dem"] = f["dem"] - f["pvc"]
        f["pvc_share"] = f["pvc"] / f["dem"]

        mm = mon.reindex(yms)
        prev_ym = np.concatenate([[yms[0]], yms[:-1]])
        for c in ["pm_mc_q60", "pm_vc_mean", "pm_vc_lng", "idx_lng_krw", "idx_coal_krw", "dubai_usd_bbl"]:
            f[c] = rows(mm[c].to_numpy())
        lng_prev = mon["pm_vc_lng"].reindex(prev_ym).to_numpy()
        q60_prev = mon["pm_mc_q60"].reindex(prev_ym).to_numpy()
        f["lng_chg_d1"] = rows(mm["pm_vc_lng"].to_numpy() / lng_prev - 1)   # 월초에만 0 이 아님
        f["q60_chg_d1"] = rows(mm["pm_mc_q60"].to_numpy() / q60_prev - 1)

        dt = pd.to_datetime(pd.Series(dates))
        f["hour"] = np.tile(np.arange(1, 25), n).astype(float)
        f["dow"] = rows(dt.dt.dayofweek)
        f["month"] = rows(dt.dt.month)
        f["dom"] = rows(dt.dt.day)
        doy = dt.dt.dayofyear.to_numpy()
        f["doy_sin"], f["doy_cos"] = rows(np.sin(2 * np.pi * doy / 365.25)), rows(np.cos(2 * np.pi * doy / 365.25))
        f["off"] = rows(off)
        f["off_prev"] = rows(np.concatenate([[0], off[:-1]]))
        f["off_next"] = rows(np.concatenate([off[1:], [0]]))
        for c in CAL_COLS:
            f[c] = h[c].to_numpy(float)

    X = pd.DataFrame(f)
    X.insert(0, "date", np.repeat(dates, 24))
    X.insert(1, "y", flat(P))
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


FEATURES_EXCLUDE = {"date", "y"}


def feature_cols(X):
    return [c for c in X.columns if c not in FEATURES_EXCLUDE]


if __name__ == "__main__":
    X = build()
    print(X.shape, X["date"].min(), X["date"].max())
    print(X.iloc[24 * 400 + 12].to_string())
