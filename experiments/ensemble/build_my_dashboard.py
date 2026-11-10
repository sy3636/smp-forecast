# -*- coding: utf-8 -*-
"""
build_my_dashboard.py — 팀 결과로 대시보드 HTML 한 장 생성 (TEAM_REPRODUCE.md 7절)
==============================================================================
  · 기간 : 성적은 config/eval_settings.json 시험 구간, 그래프·하루 자세히 보기는 전체 데이터 기간
  · 입력 : outputs/ensemble/price_predictions.csv · warnings_daily.csv · explain_daily.csv
           events_*.csv 및 warnings_*.csv — 사건 판정·대표 사례·혼동행렬에도 사용
           outputs/predict/predictions_smp_jeju.csv (있으면 제주 운영 예측, 없으면 기본 mirror 정책 적용)
  · 참고 자료 : 수요예측(D+1 하루전, MW)·태양광 계절 전망(모델 입력, features.seasonal_pv)·태양광 실적(MW, 모델 미사용)
                — common.load_hourly 에서 읽어 날짜 화면에 표시
  · 사건 판정 기준은 events.py, 가격 지표는 이 파일의 price_metrics 규칙을 따른다
  · 결과 데이터를 HTML 안에 넣으므로 더블클릭만으로 열림(서버·인터넷·외부 라이브러리 불필요)
  · 데이터가 들어간 파일이라 outputs/ 에만 저장 — 저장소에 올리지 않음

실행
  python experiments/ensemble/build_my_dashboard.py
산출
  outputs/ensemble/my_dashboard.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, SETTINGS, load_hourly, to_matrix
import events as EVM
from features import seasonal_pv

HERE = Path(__file__).resolve().parent
A, B = SETTINGS["test"]                                 # 성적 기간 = 시험 구간 (config/eval_settings.json)
V = SETTINGS["data_end"]                                # 그래프·하루 자세히 보기 끝 = 전체 기간 끝
JEJU_HISTORY = HERE.parents[1] / "outputs" / "predict" / "predictions_smp_jeju.csv"
PARTS = ["최근 가격", "전력 수요", "태양광", "요일·휴일", "연료비·원전", "계절·시간", "다른 모델 보정"]
TASKS = {"급락": "c", "급등+20%": "s20", "급등+25%": "s25", "급등+30%": "s30"}


def rd(f, end=V):
    d = pd.read_csv(OUT / f, encoding="utf-8-sig")
    return d[(d["date"] >= A) & (d["date"] <= end)]


def num(v, k=2):
    return None if pd.isna(v) else round(float(v), k)


def price_metrics(p):
    q = p.dropna(subset=["실적", "예측", "전일동시간"])
    y, f, n1 = q["실적"], q["예측"], q["전일동시간"]
    den = np.maximum(y, 20)
    r = p.dropna(subset=["실적", "하한80", "상한80"])
    ry, lo, hi = r["실적"], r["하한80"], r["상한80"]
    return {"mae": np.abs(f - y).mean(), "rmae": np.abs(f - y).mean() / np.abs(n1 - y).mean(),
            "mape": (np.abs(f - y) / den).mean() * 100, "bias": ((f - y) / den).mean() * 100,
            "cover": ((ry >= lo) & (ry <= hi)).mean() * 100, "width": (hi - lo).mean(),
            "below": (ry < lo).mean() * 100, "above": (ry > hi).mean() * 100, "n": len(q), "n_band": len(r)}


def main():
    pv_, wv, e = rd("price_predictions.csv"), rd("warnings_daily.csv"), rd("explain_daily.csv")
    pv_all = pd.read_csv(OUT / "price_predictions.csv", encoding="utf-8-sig")
    p, w = pv_[pv_["date"] <= B], wv[wv["date"] <= B]                  # 성적 계산용

    # 수요예측·태양광 (일수×24)
    h = load_hourly()
    hd = h["date"].iloc[::24].to_numpy()
    PV = to_matrix(h, "pv")
    side = {"dem": to_matrix(h, "dem"), "pvc": seasonal_pv(hd, PV), "pv": PV}
    hi_ = {d: i for i, d in enumerate(hd)}

    # 급등 문턱(시간별) = 같은 시각 직전 28일 실적 중앙값 + 20원 — events.py 와 같은 계산
    S_ = EVM.load_settings()
    act, _, _ = EVM.load_actuals(S_)
    ad = act["date"].iloc[::24].to_numpy()
    THR = EVM.baseline(act["land"].to_numpy(float).reshape(-1, 24), S_["median_days"]) + S_["spike_add_won"]
    thr_i = {d: i for i, d in enumerate(ad)}
    jeju_actual = act["jeju"].to_numpy(float).reshape(-1, 24)
    jeju_actual_by_date = {d: jeju_actual[i] for i, d in enumerate(ad)}

    jeju_forecast = {"available": False, "latest": None, "source": None, "days": []}
    if JEJU_HISTORY.exists():
        jf = pd.read_csv(JEJU_HISTORY, encoding="utf-8-sig")
        required = {"date", "hour_n", "예측"}
        missing = required - set(jf.columns)
        if missing:
            raise ValueError(f"{JEJU_HISTORY} 필수 열 누락: {', '.join(sorted(missing))}")
        jf["date"] = jf["date"].astype(str).str[:10]
        jf["hour_n"] = pd.to_numeric(jf["hour_n"], errors="raise").astype(int)
        jf["예측"] = pd.to_numeric(jf["예측"], errors="raise")
        jf = jf[jf["hour_n"].between(1, 24) & jf["예측"].notna()].drop_duplicates(
            ["date", "hour_n"], keep="last")
        jeju_forecast["days"] = [
            {"d": d,
             "p": [num(g.set_index("hour_n")["예측"].get(hour)) for hour in range(1, 25)],
             "y": [num(v) for v in jeju_actual_by_date.get(d, np.full(24, np.nan))]}
            for d, g in jf.groupby("date", sort=True)
        ]
        if jeju_forecast["days"]:
            jeju_forecast["available"] = True
            jeju_forecast["latest"] = jeju_forecast["days"][-1]["d"]
            jeju_forecast["source"] = "운영 제주 예측 이력"
    if not jeju_forecast["available"]:
        # step6_predict.py의 기본 JEJU_MODE=mirror 정책과 같이 육지 최종 예측을 제주 값으로 사용한다.
        jeju_forecast["days"] = [
            {"d": d,
             "p": [num(g.set_index("hour")["예측"].get(hour)) for hour in range(1, 25)],
             "y": [num(v) for v in jeju_actual_by_date.get(d, np.full(24, np.nan))]}
            for d, g in pv_all.groupby("date", sort=True)
        ]
        if jeju_forecast["days"]:
            jeju_forecast["available"] = True
            jeju_forecast["latest"] = jeju_forecast["days"][-1]["d"]
            jeju_forecast["source"] = "mirror 대체 예측 (육지 최종 예측을 제주에 복사)"

    # 1) 가격 — 날짜별 24시간
    days = []
    for d, g in pv_.sort_values(["date", "hour"]).groupby("date", sort=True):
        days.append({"d": d, "y": [num(v) for v in g["실적"]], "p": [num(v) for v in g["예측"]],
                     "lo": [num(v) for v in g["하한80"]], "hi": [num(v) for v in g["상한80"]],
                     "thr": [num(v, 1) for v in THR[thr_i[d]]] if d in thr_i else None,
                     **{k: [num(v, 0) for v in M[hi_[d]]] for k, M in side.items()}})

    pm = price_metrics(p)
    months = []
    for ym, g in p.groupby(p["date"].str[:7]):
        m = price_metrics(g)
        months.append({"ym": ym, "mae": num(m["mae"]), "cover": num(m["cover"], 1), "n": m["n"]})

    # 2) 경보 — 날짜별(~V) · 성적표·실제 발생일(성적 기간) · 그래프 표시용 실제 발생일(~V)
    warn, cnt, events, events_all = {}, [], {}, {}
    for r in wv.itertuples():
        k = TASKS[r.과제]
        warn.setdefault(r.date, {"dow": r.요일, "hol": None if pd.isna(r.휴일명) else r.휴일명,
                                 "m13": int(r.월초1_3일), "mid": num(r.한낮최저), "max": num(r.일최고),
                                 "maxh": None if pd.isna(r.최고시각) else int(r.최고시각), "ref": num(r.직전28일기준)})
        warn[r.date][k] = [round(float(r.확률), 4), r.단계, None if pd.isna(r.실제) else int(r.실제)]
    for task, k in [("급락", "c"), ("급등+25%", "s25")]:      # 급락 15시 · 급등 날짜 단위(전날 21시)
        g = w[w["과제"] == task].dropna(subset=["실제"])
        real = g["실제"] == 1
        row = {"task": task, "when": g["판단시각"].iloc[0], "days": len(g), "real": int(real.sum())}
        for name, on in [("alarm", g["단계"] == "경보"), ("watch", g["단계"].isin(["주의", "경보"]))]:
            row[name] = [int((on & real).sum()), int((on & ~real).sum()), int((~on & real).sum())]
        cnt.append(row)
        events[k] = g.loc[real, "date"].tolist()               # (일 단위 옛 기준 — 보조로만 남김)
        ga = wv[(wv["과제"] == task) & (wv["실제"] == 1)]
        events_all[k] = ga["date"].tolist()

    # 육지 급락(warnings_crash.py, 교체판) — 날짜 화면 확률·단계 (휴일 = c, 평일 = cw 자리에 넣음)
    if (OUT / "warnings_crash.csv").exists():
        for r in rd("warnings_crash.csv").itertuples():
            if r.date in warn and isinstance(r.단계, str):
                v = [round(float(r.확률), 4), r.단계, None if pd.isna(r.실제) else int(r.실제)]
                warn[r.date].pop("c", None); warn[r.date].pop("cw", None)
                warn[r.date]["c" if r.구분 == "휴일" else "cw"] = v

    # 육지 급등(warnings_spike.py, 새 기준 학습) — 날짜 화면 확률·단계를 이것으로 바꿈 (일 최고·28일 기준은 보조 줄용으로 남김)
    if (OUT / "warnings_spike.csv").exists():
        for r in rd("warnings_spike.csv").itertuples():
            if r.date in warn and isinstance(r.단계, str):
                warn[r.date]["s25"] = [round(float(r.확률), 4), r.단계, None if pd.isna(r.실제) else int(r.실제)]

    # 제주 급락(warnings_jeju.py) — 날짜 화면 확률·단계
    jrep = None
    jf = "warnings_jeju_v2.csv" if (OUT / "warnings_jeju_v2.csv").exists() else "warnings_jeju.csv"   # v2 우선
    if (OUT / jf).exists():
        for r in rd(jf).itertuples():
            if r.date in warn and isinstance(r.단계, str):
                warn[r.date]["j"] = [round(float(r.확률), 4), r.단계, None if pd.isna(r.실제) else int(r.실제)]
        jr = json.load(open(OUT / "warnings_jeju_report.json", encoding="utf-8"))
        jrep = {"w": jr["chosen"]["weights"], "mode": jr["threshold_mode"], "note": jr["note"]}

    # 평일 급락(warnings_weekday.py) — 날짜 화면에만 표시, 위 성적표·대표 숫자에는 넣지 않음
    wk = None
    if (OUT / "warnings_weekday.csv").exists():
        for r in rd("warnings_weekday.csv").itertuples():
            if r.date in warn:
                warn[r.date]["cw"] = [round(float(r.확률), 4), r.단계, None if pd.isna(r.실제) else int(r.실제)]
        rep = json.load(open(OUT / "warnings_weekday_report.json", encoding="utf-8"))
        wk = {"w": rep["weights"], "watch": round(rep["th_watch"], 3), "alarm": round(rep["th_alarm"], 3),
              "dev": rep["dev_weekday"], "test": rep["test_weekday"]}

    # 3) 과제 3 근거
    expl = {r["date"]: {"ref": num(r["ref"]), "ens": num(r["ens"]), "parts": [num(r[c]) for c in PARTS],
                        "s": r["sentence"]} for _, r in e.iterrows()}
    es = e[e["date"] <= B]
    impact = es[PARTS[:6]].abs().mean().sort_values(ascending=False).round(1)
    check = float((es["ref"] + es[PARTS].sum(axis=1) - es["ens"]).abs().max())

    # 4) 시간 구간 사건 (events.py) — 성적표·카드·월별·날짜 화면
    ev = json.load(open(OUT / "events_report.json", encoding="utf-8"))
    sg = pd.read_csv(OUT / "events_segments.csv", encoding="utf-8-sig")
    dl = pd.read_csv(OUT / "events_daily.csv", encoding="utf-8-sig")
    seg = {}
    for r in sg[(sg["date"] >= A) & (sg["date"] <= V)].itertuples():
        seg.setdefault(r.date, []).append([r.key, int(r.시작시), int(r.끝시), r.가격모델, r.경보단계, r.경보모델,
                                           num(r.실적평균), num(r.예측평균), None if pd.isna(r.기준값평균) else num(r.기준값평균)])
    inr = sg[(sg["date"] >= A) & (sg["date"] <= V)]
    ev_days = {"c": sorted(inr.loc[(inr["key"] == "crash") & (inr["구분"] == "휴일"), "date"].unique().tolist()),
               "cw": sorted(inr.loc[(inr["key"] == "crash") & (inr["구분"] == "평일"), "date"].unique().tolist()),
               "s": sorted(inr.loc[inr["key"] == "spike", "date"].unique().tolist()),
               "j": sorted(inr.loc[inr["key"] == "jeju", "date"].unique().tolist())}
    sb = sg[(sg["date"] >= A) & (sg["date"] <= B)]
    for mo in months:                                          # 월별 [적중 구간, 실제 구간] — 급락 주의 이상 · 급등 경보
        g = sb[sb["date"].str[:7] == mo["ym"]]
        c, s_ = g[g["key"] == "crash"], g[g["key"] == "spike"]
        mo["c"] = [int(c["경보모델"].isin(["성공", "부분"]).sum()), len(c)]
        mo["s25"] = [int((s_["경보모델"] == "성공").sum()), len(s_)]
    alld = dl[dl["판정"] & (dl["date"] >= A) & (dl["date"] <= B)]          # 시험 구간만
    # 학습·전체 성적은 화면에 넣지 않음 (events_report.json 에만)
    evinfo = {"periods": {"test": ev["periods"]["test"]}, "set": {k: SETTINGS[k] for k in ("train", "test", "data_start", "data_end", "issue_hour",
                                                                        "spike_add_won", "land_crash_won", "land_crash_hours",
                                                              "median_days", "jeju_rule", "second_test_from")}, "jeju": jrep,
              "judgeStart": ev["settings"]["judge_start"], "checks": ev["checks"],
              "spikeDays": int((alld["급등구간"] > 0).sum()), "judgeDays": len(alld),
              "spikeSegs": int(((sg["key"] == "spike") & (sg["date"] >= A) & (sg["date"] <= B)).sum()),
              "auxDays": int(alld["보조_일급등x1.25"].sum()),
              "aux": {d: True for d in dl.loc[dl["보조_일급등x1.25"] & (dl["date"] >= A), "date"]}}

    # 발표용 「사건 판정 결과」 — 화면에서 시험 기간을 바꾸면 다시 세도록 판정일 전체(판정 시작~끝)의 구간·헛경보를 넣음
    fa = pd.read_csv(OUT / "events_false_alarms.csv", encoding="utf-8-sig")
    judge_seg = [[r.date, r.key, r.구분, r.가격모델, r.경보모델, int(r.시작시), int(r.끝시),
                  num(r.실적평균), num(r.예측평균), num(r.문턱)] for r in sg.itertuples()]

    # 날짜 단위 혼동행렬용 실제 사건·경보·가격 예측. 예측이 없는 날짜는 None으로 두어 채점에서 제외한다.
    daily_judge = {}
    for r in dl[dl["판정"]].itertuples():
        daily_judge[r.date] = [r.구분, int(r.급등구간 > 0), int(r.급락구간 > 0), int(r.제주0원구간 > 0),
                               None, None, None, None, None, None]

    warning_sources = {
        "spike": ("warnings_spike.csv", SETTINGS["issue_hour"]["spike"]),
        "crash": ("warnings_crash.csv", SETTINGS["issue_hour"]["crash"]),
        "jeju": (jf, SETTINGS["issue_hour"]["crash"]),
    }
    prob_index = {"spike": 4, "crash": 5, "jeju": 6}
    for key, (filename, hour) in warning_sources.items():
        path = OUT / filename
        if not path.exists():
            continue
        source = pd.read_csv(path, encoding="utf-8-sig")
        source = source[source["판단시각"] == f"{hour}시"]
        for r in source.itertuples():
            row = daily_judge.get(r.date)
            if row is not None and isinstance(r.단계, str):
                row[prob_index[key]] = int(r.단계 in ("주의", "경보"))

    price_index = {"spike": 7, "crash": 8}
    hours_by_key = {"spike": list(range(1, 25)), "crash": S_["land_crash_hours"]}
    for d, g in pv_all.groupby("date", sort=False):
        if d not in daily_judge or d not in thr_i:
            continue
        forecasts = g.set_index("hour")["예측"]
        for key, hours in hours_by_key.items():
            if not all(h in forecasts.index and pd.notna(forecasts.loc[h]) for h in hours):
                continue
            values = np.asarray([float(forecasts.loc[h]) for h in hours])
            if key == "spike":
                thresholds = THR[thr_i[d], np.asarray(hours) - 1]
                if np.isnan(thresholds).any():
                    continue
                predicted = bool((values >= thresholds).any())
            else:
                predicted = bool((values < S_["land_crash_won"]).any())
            daily_judge[d][price_index[key]] = int(predicted)

    for day in jeju_forecast["days"]:
        row = daily_judge.get(day["d"])
        if row is None or any(v is None for v in day["p"]):
            continue
        values = np.asarray(day["p"], dtype=float)
        predicted = np.any(values <= 0) if S_["jeju_rule"] == "le0" else np.any(values == 0)
        row[9] = int(predicted)

    judge_days = [[d, *values] for d, values in sorted(daily_judge.items())]
    judge = {"seg": judge_seg,
             "days": judge_days,
             "fa": [[r.모델, r.key, r.date, r.구분, "" if pd.isna(r.단계) else r.단계] for r in fa.itertuples()]}

    data = {"period": [A, V], "score": [A, B], "ev": evinfo, "judge": judge, "seg": seg, "evDays": ev_days, "days": days, "months": months, "warn": warn, "cnt": cnt,
            "events": events, "eventsAll": events_all, "wk": wk, "expl": expl, "parts": PARTS, "impact": [[k, float(v)] for k, v in impact.items()],
            "jejuForecast": jeju_forecast, "explCheck": round(check, 2), "pm": {k: (round(float(v), 4) if k not in ("n", "n_band") else int(v))
                                                for k, v in pm.items()}}
    html = (HERE / "my_dashboard_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "my_dashboard.html").write_text(html, encoding="utf-8")

    T = ev["periods"]["test"]
    ch, cw_, sp = T["crash"]["휴일"], T["crash"]["평일"], T["spike"]
    sw = {j: sp["휴일"]["경보"][j] + sp["평일"]["경보"][j] for j in ("성공", "부분", "실패")}
    print(f"저장 : {OUT / 'my_dashboard.html'}  ({len(html) / 1e6:.2f}MB, {len(days)}일 {A}~{V} · 성적 {A}~{B} {T['name']})")
    print(f"1. 급락(휴일 구간) : 경보 {ch['경보']['성공']}/{ch['구간']} · 주의 이상 {ch['경보']['성공'] + ch['경보']['부분']}/{ch['구간']} "
          f"(헛경보 경보 {ch['헛경보_경보일']}일·주의 {ch['헛경보_주의일']}일) · 평일 {cw_['경보']['성공']}/{cw_['구간']}")
    print(f"2. 가격예측 : {pm['mae']:.2f}원 (평균 오차 MAE)")
    print(f"3. 급등(구간) : 경보 {sw['성공']}/{sp['휴일']['구간'] + sp['평일']['구간']} · 주의 {sw['부분']} · 놓침 {sw['실패']} "
          f"(헛경보 경보 {sp['휴일']['헛경보_경보일'] + sp['평일']['헛경보_경보일']}일)")
    print(f"MAE {pm['mae']:.2f} · rMAE {pm['rmae']:.4f} · MAPE {pm['mape']:.2f}% · 편의 {pm['bias']:+.2f}% · "
          f"포함률 {pm['cover']:.1f}% · 폭 {pm['width']:.1f}원 · 아래 {pm['below']:.1f}% · 위 {pm['above']:.1f}%")
    print(f"근거 {len(e)}일 · 검산 최대 오차 {check:.2f}원")


if __name__ == "__main__":
    main()
