# -*- coding: utf-8 -*-
"""
events.py — 시간별 사건(육지 급등·육지 급락·제주 0원) 판정과 모델 채점 (2026-11-08)
==============================================================================
설정 : config/eval_settings.json (기간·문턱·예측 시점·연속 구간 규칙 — 하드코딩하지 않음)

1) 사건 정의 (거래시간 1~24시, h시 = 실제 (h-1):00~h:00)
   · 육지 급등 : 실적 ≥ 기준값 + spike_add_won(20원)
                기준값 = 판정일 D의 h시에 대해 D-28 ~ D-1 각 날 h시 실적의 중앙값 (당일 제외, 같은 시각끼리)
   · 육지 급락 : 거래시간 land_crash_hours(11~15시) 중 실적 < land_crash_won(30원)
   · 제주 급락 : (include_jeju=false 면 계산 안 함 — 기본 끔) 제주 SMP 0원 (jeju_rule: "le0" = 0 이하(기본) · "eq0" = 정확히 0)
                음수 SMP(최저 −79원)가 있어 「0 이하」로 세야 검증표와 같음 — 음수 시간이 0원 시간을 이어 구간 수가 바뀜
   · 조건을 만족하는 시간이 이어지면 1구간. segment_cross_midnight=false 이면 거래일 안에서만 묶음.
   · 모든 구간에 휴일(주말+공휴일)/평일 표시
   · 보조 지표(원인 분석용) : 일 최고 ≥ 직전 28일 일 최고 중앙값 × spike_daily_ratio(1.25) — events_daily.csv
2) 채점
   · 가격 모델(시간별 예측) : 구간 시간의 절반 이상에서 예측이 문턱을 넘으면 성공,
     아니면 구간 평균으로 (예측−기준값) ≥ (실적−기준값)/2 (급락은 (기준값−예측) ≥ (기준값−실적)/2) 이면 부분, 그 외 실패
     헛경보 = 예측은 문턱을 넘었는데 실적은 넘지 않은 시간의 연속 구간 수
   · 확률 모델(주의·경보) : 그날 경보 = 성공, 주의 = 부분, 없음 = 실패 (하루 경보를 그날 모든 구간에 적용)
     헛경보 = 경보/주의를 냈는데 그날 그 사건이 없던 날 수 (주의·경보 따로)
   · 전날 issue_hour 시에 만든 예측만 채점. 생성 시각(made_at)이 있으면 그것으로, 없으면 「전날 + 판단 시각」으로 검사해
     판정일 0시 이후에 만든 예측은 뺀다.
3) 데이터 : 2021-01-01~ 교육용 smp_dataset_hourly.csv, 2020년 공공데이터포털 SMP API(api_smp_demand_hourly.csv),
   데이터셋 이후는 api_smp_recent.csv, 빠진 날(2026-07-31)은 api_smp_fill.csv. 겹치는 시간은 값이 같은지 검사.
   공휴일 : 특일정보 API(api_holiday.csv) + 주말.

실행 : python experiments/ensemble/events.py            (판정·채점·검증표 출력)
산출 : outputs/ensemble/events_segments.csv · events_false_alarms.csv · events_daily.csv · events_report.json
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import BASE, DATA, OUT, rd

SETTINGS = BASE / "config" / "eval_settings.json"
TYPES = {"spike": "육지 급등", "crash": "육지 급락", "jeju": "제주 0원"}

# 검증표 (실적만으로 센 구간 수, 휴일 / 평일) — 2020-01-29 ~ 2026-10-31
EXPECTED = {
    "spike": {2020: (2, 17), 2021: (11, 35), 2022: (71, 259), 2023: (43, 180), 2024: (48, 279), 2025: (54, 239), 2026: (55, 190)},
    "crash": {2020: (0, 0), 2021: (0, 0), 2022: (2, 0), 2023: (0, 0), 2024: (8, 0), 2025: (18, 1), 2026: (18, 2)},
    "jeju": {2020: (0, 0), 2021: (0, 0), 2022: (0, 0), 2023: (0, 0), 2024: (10, 16), 2025: (29, 30), 2026: (23, 26)},
}
REQUIRED_2020_HOLIDAYS = ["2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-04-15", "2020-08-17"]


# ------------------------------------------------------------------ 설정
def load_settings():
    S = json.loads(SETTINGS.read_text(encoding="utf-8"))
    S["judge_start"] = (date.fromisoformat(S["data_start"]) + timedelta(days=S["median_days"])).isoformat()
    return S


def end_excl(d):
    """양끝 포함 종료일 → 다음날(구간 [a, b) 표기용)"""
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()


def period_label(S, a, b):
    """시험 구간 이름 — 설정 test_label 이 있으면 그 이름(2026-11-08부터 「시험」), 없으면 2026년 포함 시 「2차 시험」"""
    if S.get("test_label"):
        return S["test_label"]
    return "2차 시험" if b >= S["second_test_from"] else "시험"


# ------------------------------------------------------------------ 데이터
def _cmp(a, b, name, checks):
    """겹치는 시간의 값 일치 검사"""
    m = a.merge(b, on=["date", "hour"], suffixes=("_a", "_b"))
    if m.empty:
        checks.append({"검사": name, "겹친 시간": 0, "다른 시간": 0, "결과": "겹침 없음"})
        return
    bad = 0
    for c in ("land", "jeju"):
        x, y = m[c + "_a"], m[c + "_b"]
        bad_c = ~(np.isclose(x, y, atol=1e-6) | (x.isna() & y.isna()))
        bad += int(bad_c.sum())
    checks.append({"검사": name, "겹친 시간": len(m), "다른 시간": bad, "결과": "일치" if bad == 0 else "불일치"})


def load_actuals(S):
    """시간별 실적 SMP(육지·제주) — 세 소스를 합치고 겹침을 검사. 반환: (날짜×시간 표, 검사 목록)"""
    cols = {"hour_n": "hour", "smp_land": "land", "smp_jeju": "jeju"}
    edu = rd(DATA / "processed" / "smp_dataset_hourly.csv", usecols=["date", "hour_n", "smp_land", "smp_jeju"]).rename(columns=cols)
    edu = edu[(edu["date"] >= "2021-01-01") & edu["land"].notna()]
    api = rd(DATA / "interim" / "api_smp_demand_hourly.csv", usecols=["date", "hour_n", "smp_land", "smp_jeju"]).rename(columns=cols)
    rec = rd(DATA / "interim" / "api_smp_recent.csv", usecols=["date", "hour_n", "smp_land", "smp_jeju"]).rename(columns=cols)
    fill_p = DATA / "interim" / "api_smp_fill.csv"      # 빠진 날 보충(같은 SMP API) — 2026-07-31
    fill = rd(fill_p, usecols=["date", "hour_n", "smp_land", "smp_jeju"]).rename(columns=cols) if fill_p.exists() else edu.iloc[:0]
    checks = []
    _cmp(api[api["date"].str[:7] == "2021-01"], edu[edu["date"].str[:7] == "2021-01"], "2020 API ↔ 교육용 (2021-01)", checks)
    _cmp(rec, edu, "최근 API ↔ 교육용", checks)
    _cmp(fill, edu, "보충 API ↔ 교육용", checks)
    last = edu["date"].max()
    parts = [api[api["date"] < "2021-01-01"], edu, rec[rec["date"] > last], fill]   # 앞 소스 우선, 보충은 빈 시간만
    a = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "hour"])
    days = pd.date_range(S["data_start"], S["data_end"], freq="D").strftime("%Y-%m-%d")
    g = pd.MultiIndex.from_product([days, range(1, 25)], names=["date", "hour"]).to_frame(index=False)
    g = g.merge(a, on=["date", "hour"], how="left")
    return g, checks, {"2020": "api_smp_demand_hourly.csv", "2021~" + last: "smp_dataset_hourly.csv",
                       "이후": "api_smp_recent.csv", "빠진 날 보충": "api_smp_fill.csv " + ",".join(sorted(fill["date"].unique()))}


def load_holidays(S, days):
    """휴일 = 주말 + 특일정보 API 공휴일. 2020 필수 공휴일 확인, 2021~ 데이터셋 휴일 표시와 비교."""
    hol = rd(DATA / "interim" / "api_holiday.csv")
    hol = hol[hol["is_holiday"] == 1].drop_duplicates("date").set_index("date")["date_name"]
    dt = pd.to_datetime(pd.Series(days))
    off = (dt.dt.dayofweek >= 5).to_numpy() | pd.Series(days).isin(hol.index).to_numpy()
    missing = [d for d in REQUIRED_2020_HOLIDAYS if d not in hol.index]
    ds = rd(DATA / "processed" / "smp_dataset_hourly.csv", usecols=["date", "hour_n", "is_holiday"])
    ds = ds[(ds["hour_n"] == 1) & (ds["date"] >= "2021-01-01")].set_index("date")["is_holiday"]
    api_flag = pd.Series(ds.index.isin(hol.index).astype(int), index=ds.index)
    diff = ds[ds.astype(int) != api_flag].index.tolist()
    return off, hol, {"2020 필수 공휴일 누락": missing, "2021~ 데이터셋과 다른 날": diff}


# ------------------------------------------------------------------ 사건
def baseline(M, k):
    """같은 시각끼리 직전 k일(당일 제외) 중앙값. 앞 k일은 NaN."""
    n = M.shape[0]
    med = np.full(M.shape, np.nan)
    for i in range(k, n):
        med[i] = np.nanmedian(M[i - k:i], axis=0)
    return med


def runs(flag, cross_midnight):
    """연속 True 구간 → [(날짜 index, 시작시 1~24, 끝시)]. cross_midnight=False 면 날짜 안에서만."""
    out = []
    if not cross_midnight:
        for i, row in enumerate(flag):
            h = 0
            while h < 24:
                if row[h]:
                    s = h
                    while h < 24 and row[h]:
                        h += 1
                    out.append((i, s + 1, h))
                else:
                    h += 1
    else:
        f = flag.ravel()
        k = 0
        while k < len(f):
            if f[k]:
                s = k
                while k < len(f) and f[k]:
                    k += 1
                out.append((s // 24, s % 24 + 1, (k - 1) % 24 + 1 + 24 * ((k - 1) // 24 - s // 24)))
            else:
                k += 1
    return out


def event_flags(S, L, J, med):
    h = np.arange(1, 25)
    a, b = S["land_crash_hours"]
    win = (h >= a) & (h <= b)
    with np.errstate(invalid="ignore"):
        spike = L >= med + S["spike_add_won"]
        crash = (L < S["land_crash_won"]) & win
        jeju = (J == 0) if S["jeju_rule"] == "eq0" else (J <= 0)
    return {"spike": spike & ~np.isnan(L) & ~np.isnan(med), "crash": crash & ~np.isnan(L), "jeju": jeju & ~np.isnan(J)}


# ------------------------------------------------------------------ 예측
def issue_ok(dates, issue_hour, made_at=None):
    """전날 issue_hour 시에 만든 예측만 True. made_at 이 있으면 그것을, 없으면 전날+판단시각을 생성 시각으로 본다."""
    d0 = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    made = pd.to_datetime(pd.Series(made_at)).reset_index(drop=True) if made_at is not None else \
        d0 - pd.Timedelta(days=1) + pd.Timedelta(hours=issue_hour)
    return (made < d0).to_numpy()


def load_price(S, days):
    """가격 모델 시간별 예측 (날짜×24). 전날 issue_hour.price 시 예측만."""
    p = pd.read_pickle(OUT / "price_final.pkl")
    ok = issue_ok(p["date"], S["issue_hour"]["price"], p["made_at"] if "made_at" in p.columns else None)
    p = p[ok]
    idx = {d: i for i, d in enumerate(days)}
    P = np.full((len(days), 24), np.nan)
    p = p[p["date"].isin(idx)]
    P[p["date"].map(idx).to_numpy(), p["hour"].to_numpy() - 1] = p["pred"].to_numpy()
    return P


def load_levels(S, days):
    """확률 모델 일 단위 단계 {사건: 날짜별 '경보'/'주의'/'정상'/''}. 급락 = 휴일(warnings_final) + 평일(warnings_weekday)."""
    w = pd.read_pickle(OUT / "warnings_final.pkl")
    lv = {}
    for k, label, hour in [("crash", "crash", S["issue_hour"]["crash"]), ("spike", "spike25", S["issue_hour"]["spike"])]:
        x = w[(w["label"] == label) & (w["stage"] == str(hour)) & (w["level"] != "")]
        x = x[issue_ok(x["date"], hour)]
        s = x.set_index("date")["level"]
        if k == "crash":
            f = OUT / "warnings_weekday.csv"
            if f.exists():
                wk = rd(f)
                wk = wk[wk["판단시각"] == f"{hour}시"]
                wk = wk[issue_ok(wk["date"], hour)]
                s = pd.concat([s, wk.set_index("date")["단계"]])
                s = s[~s.index.duplicated()]
        lv[k] = pd.Series(days).map(s).fillna("").to_numpy()
    f = OUT / "warnings_crash.csv"                     # 육지 급락 경보 — 교체판(warnings_crash.py)이 있으면 그것을 씀 (휴일만 경보, 한 단계)
    if f.exists():
        hour = S["issue_hour"]["crash"]
        wc = rd(f)
        wc = wc[(wc["판단시각"] == f"{hour}시") & wc["단계"].notna()]
        wc = wc[issue_ok(wc["date"], hour)]
        lv["crash"] = pd.Series(days).map(wc.set_index("date")["단계"]).fillna("").to_numpy()
    f = OUT / "warnings_spike.csv"                     # 육지 급등 경보 — 새 기준으로 학습한 모델(warnings_spike.py)이 있으면 그것을 씀
    if f.exists():
        hour = S["issue_hour"]["spike"]
        ws = rd(f)
        ws = ws[(ws["판단시각"] == f"{hour}시") & ws["단계"].notna()]
        ws = ws[issue_ok(ws["date"], hour)]
        lv["spike"] = pd.Series(days).map(ws.set_index("date")["단계"]).fillna("").to_numpy()
    f = OUT / "warnings_jeju_v2.csv"                   # 제주 급락 경보 — v2(warnings_jeju_v2.py)가 있으면 v2, 없으면 v1(warnings_jeju.py)
    if not f.exists():
        f = OUT / "warnings_jeju.csv"
    if f.exists():
        hour = S["issue_hour"]["crash"]
        wj = rd(f)
        wj = wj[(wj["판단시각"] == f"{hour}시") & wj["단계"].notna()]
        wj = wj[issue_ok(wj["date"], hour)]
        lv["jeju"] = pd.Series(days).map(wj.set_index("date")["단계"]).fillna("").to_numpy()
    return lv


# ------------------------------------------------------------------ 채점
def judge_price(kind, act, pred, med, S):
    if np.isnan(pred).any():
        return "예측 없음"
    if kind == "spike":
        thr = med + S["spike_add_won"]
        if (pred >= thr).sum() * 2 >= len(pred):
            return "성공"
        return "부분" if np.mean(pred - med) >= np.mean(act - med) / 2 else "실패"
    if (pred < S["land_crash_won"]).sum() * 2 >= len(pred):
        return "성공"
    return "부분" if np.mean(med - pred) >= np.mean(med - act) / 2 else "실패"


LEVEL_JUDGE = {"경보": "성공", "주의": "부분", "정상": "실패", "": "예측 없음"}


def main():
    S = load_settings()
    A, checks, sources = load_actuals(S)
    days = A["date"].iloc[::24].to_numpy()
    L, J = A["land"].to_numpy(float).reshape(-1, 24), A["jeju"].to_numpy(float).reshape(-1, 24)
    off, hol, hol_check = load_holidays(S, days)
    med = baseline(L, S["median_days"])
    judge = days >= S["judge_start"]
    F = event_flags(S, L, J, med)
    if not S.get("include_jeju", True):                # 제주는 설정으로 끔(2026-11-08 기본 끔)
        F.pop("jeju")
    KEYS = list(F)
    for k in F:
        F[k] &= judge[:, None]
    P = load_price(S, days)
    LV = load_levels(S, days)

    # 1) 사건 구간 + 채점
    seg = []
    for k, flag in F.items():
        for i, h1, h2 in runs(flag, S["segment_cross_midnight"]):
            hs = np.arange(h1, h2 + 1) - 1
            hs_day = hs % 24
            rows = (i + hs // 24)
            act, pr, md = L[rows, hs_day] if k != "jeju" else J[rows, hs_day], P[rows, hs_day], med[rows, hs_day]
            r = {"사건": TYPES[k], "key": k, "date": days[i], "구분": "휴일" if off[i] else "평일",
                 "휴일명": hol.get(days[i], ""), "시작시": h1, "끝시": h2, "시간수": len(hs),
                 "실적평균": round(float(np.nanmean(act)), 2), "기준값평균": round(float(np.nanmean(md)), 2) if k == "spike" else None,
                 "문턱": (round(float(np.nanmean(md)) + S["spike_add_won"], 2) if k == "spike" else
                        (S["land_crash_won"] if k == "crash" else 0)),
                 "예측평균": None if np.isnan(pr).all() else round(float(np.nanmean(pr)), 2)}
            if k in ("spike", "crash"):
                r["가격모델"] = judge_price(k, act, pr, md if k == "spike" else md, S)
                r["경보단계"] = LV[k][i] or "없음"
                r["경보모델"] = LEVEL_JUDGE[LV[k][i]]
            else:                                      # 제주: 시간별 가격 모델 없음, 경보 모델만 채점
                r["가격모델"] = "모델 없음"
                lvj = LV["jeju"][i] if "jeju" in LV else ""
                r["경보단계"] = lvj or "없음"
                r["경보모델"] = LEVEL_JUDGE[lvj] if "jeju" in LV else "모델 없음"
            seg.append(r)
    seg = pd.DataFrame(seg).sort_values(["date", "key", "시작시"]).reset_index(drop=True)

    # 2) 헛경보 — 가격 모델(시간 구간) · 확률 모델(날짜)
    fa = []
    h = np.arange(1, 25)
    a_, b_ = S["land_crash_hours"]
    with np.errstate(invalid="ignore"):
        pf = {"spike": (P >= med + S["spike_add_won"]) & ~F["spike"],
              "crash": (P < S["land_crash_won"]) & ((h >= a_) & (h <= b_)) & ~F["crash"]}
    has_act = ~np.isnan(L)
    for k, flag in pf.items():
        flag = flag & has_act & judge[:, None] & ~np.isnan(med if k == "spike" else L)
        for i, h1, h2 in runs(flag, S["segment_cross_midnight"]):
            fa.append({"모델": "가격", "사건": TYPES[k], "key": k, "date": days[i], "구분": "휴일" if off[i] else "평일",
                       "단계": "", "시작시": h1, "끝시": h2})
    ev_day = {k: F[k].any(axis=1) for k in F}
    for k in [k for k in ("spike", "crash", "jeju") if k in LV and k in F]:
        for i in np.where(judge & ~ev_day[k] & np.isin(LV[k], ["경보", "주의"]) & has_act.any(axis=1))[0]:
            fa.append({"모델": "경보", "사건": TYPES[k], "key": k, "date": days[i], "구분": "휴일" if off[i] else "평일",
                       "단계": LV[k][i], "시작시": None, "끝시": None})
    fa = pd.DataFrame(fa, columns=["모델", "사건", "key", "date", "구분", "단계", "시작시", "끝시"])

    # 3) 날짜 표 + 보조 지표(일 최고 × 1.25)
    dmax = np.nanmax(np.where(np.isnan(L), -np.inf, L), axis=1)
    dmax[np.isinf(dmax)] = np.nan
    ref = np.full(len(days), np.nan)
    for i in range(S["median_days"], len(days)):
        ref[i] = np.nanmedian(dmax[i - S["median_days"]:i])
    daily = pd.DataFrame({"date": days, "구분": np.where(off, "휴일", "평일"), "휴일명": [hol.get(d, "") for d in days],
                          "판정": judge, "급등구간": [0] * len(days), "급락구간": [0] * len(days),
                          **({"제주0원구간": [0] * len(days)} if "jeju" in KEYS else {}),
                          "일최고": dmax, "직전28일일최고중앙값": ref,
                          "보조_일급등x1.25": (dmax >= ref * S["spike_daily_ratio"]) & judge & ~np.isnan(ref)})
    for k, c in [("spike", "급등구간"), ("crash", "급락구간"), ("jeju", "제주0원구간")][:len(KEYS)]:
        cnt = seg[seg["key"] == k].groupby("date").size()
        daily[c] = daily["date"].map(cnt).fillna(0).astype(int)

    # 4) 기간별 요약
    def summarize(a, b):
        s = seg[(seg["date"] >= a) & (seg["date"] <= b)]
        f = fa[(fa["date"] >= a) & (fa["date"] <= b)]
        dj = daily[(daily["date"] >= a) & (daily["date"] <= b) & daily["판정"]]
        out = {"판정일": int(len(dj))}
        for k in KEYS:
            o = {}
            for g in ("휴일", "평일"):
                x = s[(s["key"] == k) & (s["구분"] == g)]
                e = {"구간": int(len(x)), "시간": int(x["시간수"].sum())}
                if k != "jeju" or "jeju" in LV:
                    for m, col in [("가격", "가격모델"), ("경보", "경보모델")]:
                        vc = x[col].value_counts()
                        e[m] = {j: int(vc.get(j, 0)) for j in ("성공", "부분", "실패", "예측 없음")}
                    if k == "jeju":
                        e["가격"] = None                       # 제주 시간별 가격 모델 없음
                    fx = f[(f["key"] == k) & (f["구분"] == g)]
                    e["헛경보_가격구간"] = int((fx["모델"] == "가격").sum()) if k != "jeju" else None
                    e["헛경보_경보일"] = int(((fx["모델"] == "경보") & (fx["단계"] == "경보")).sum())
                    e["헛경보_주의일"] = int(((fx["모델"] == "경보") & (fx["단계"] == "주의")).sum())
                o[g] = e
            o["사건일"] = int((dj[{"spike": "급등구간", "crash": "급락구간", "jeju": "제주0원구간"}[k]] > 0).sum())
            out[k] = o
        return out

    tr, te = S["train"], S["test"]
    report = {"settings": {k: v for k, v in S.items() if not k.startswith("_")},
              "periods": {"train": {"name": "학습", "range": tr, **summarize(*tr)},
                          "test": {"name": period_label(S, *te), "range": te, **summarize(*te)},
                          "all": {"name": "전체", "range": [S["judge_start"], S["data_end"]], **summarize(S["judge_start"], S["data_end"])}},
              "checks": checks, "holiday_check": hol_check, "sources": sources}
    # 연도별 (검증표)
    yr = {}
    for y in range(int(S["judge_start"][:4]), int(S["data_end"][:4]) + 1):
        a, b = max(S["judge_start"], f"{y}-01-01"), min(S["data_end"], f"{y}-12-31")
        s = seg[(seg["date"] >= a) & (seg["date"] <= b)]
        yr[y] = {k: (int(((s["key"] == k) & (s["구분"] == "휴일")).sum()), int(((s["key"] == k) & (s["구분"] == "평일")).sum()))
                 for k in KEYS}
    report["by_year"] = {str(y): v for y, v in yr.items()}

    seg.to_csv(OUT / "events_segments.csv", index=False, encoding="utf-8-sig")
    fa.to_csv(OUT / "events_false_alarms.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "events_daily.csv", index=False, encoding="utf-8-sig")
    (OUT / "events_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ---------- 출력
    print(f"설정: 전체 {S['data_start']}~{S['data_end']} · 판정 시작 {S['judge_start']} · 학습 {tr[0]}~{tr[1]} · "
          f"{period_label(S, *te)} {te[0]}~{te[1]} · 예측 시점 전날 가격 {S['issue_hour']['price']}시 · "
          f"급락 {S['issue_hour']['crash']}시 · 급등 {S['issue_hour']['spike']}시")
    for c in checks:
        print(f"[겹침 검사] {c['검사']}: {c['겹친 시간']}시간 중 다른 값 {c['다른 시간']} → {c['결과']}")
    print(f"[공휴일] 2020 필수 누락 {hol_check['2020 필수 공휴일 누락'] or '없음'} · 2021~ 데이터셋과 다른 날 {hol_check['2021~ 데이터셋과 다른 날'] or '없음'}")
    nd = int(judge.sum())
    print(f"\n판정일 {nd:,}일 — 연도별 구간 수 (휴일 / 평일), 괄호 = 검증표와 차이")
    print(f"{'연도':<6}" + "".join(f"{TYPES[k]:>22}" for k in KEYS))
    tot = {k: [0, 0] for k in KEYS}
    for y, v in yr.items():
        line = f"{y:<6}"
        for k in KEYS:
            h_, w_ = v[k]
            eh, ew = EXPECTED[k].get(y, (None, None))
            tot[k][0] += h_; tot[k][1] += w_
            dd = "" if eh is None else f"({h_ - eh:+d}/{w_ - ew:+d})"
            line += f"{f'{h_} / {w_} {dd}':>24}"
        print(line)
    line = f"{'합계':<6}"
    for k in KEYS:
        eh = sum(v[0] for v in EXPECTED[k].values()); ew = sum(v[1] for v in EXPECTED[k].values())
        line += f"{f'{tot[k][0]:,} / {tot[k][1]:,} ({tot[k][0] - eh:+d}/{tot[k][1] - ew:+d})':>24}"
    print(line)
    sd = int((daily["급등구간"][daily["판정"]] > 0).sum())
    print(f"육지 급등 합계 {sum(tot['spike']):,}구간 (참고 1,483) · 급등 있는 날 {sd}일 = 판정일의 {sd / nd * 100:.1f}% (참고 833일 · 34%)")
    print(f"보조 지표(일 최고 × {S['spike_daily_ratio']}) 해당일 {int(daily['보조_일급등x1.25'].sum())}일")
    return report


if __name__ == "__main__":
    main()
