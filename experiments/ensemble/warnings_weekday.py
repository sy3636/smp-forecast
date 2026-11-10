# -*- coding: utf-8 -*-
"""
warnings_weekday.py — 평일 급락 경보 (대시보드 날짜 화면용, 기존 휴일·주말 급락 결과는 그대로 둠)
==============================================================================
  · 라벨 : 11~15시 육지 SMP 최저 < 30원/kWh (warnings_model.py 와 같음 — 다만 평일도 셈)
  · 평일만 따로 학습하지 않음 : 평일 급락은 2021~2025년에 1일(2025-11-26)뿐이라
    개발 구간에서 비율·「80%·95% 잡는 값」을 정할 수 없음
    → 휴일·주말 + 평일 「전체 날짜」로 같은 성분(로지스틱·LightGBM 분류·가격신호)을 매달 재학습하고,
      비율·주의/경보 기준은 개발 구간(2020~2024, 2026-11-08 변경) 전체 날짜 성적으로 결정, 2025~ 시험은 평가만
  · 판단 시각 : 전날 15시 (warnings_model.py 의 daily_table("15") — 자체 수요 추정·수요 없는 Chronos)
  · 대표 숫자(18/18 등)·warnings_daily.csv·warnings_final.pkl·warnings_report.json 은 건드리지 않음

실행 (warnings_model.py --final 이후)
  python experiments/ensemble/warnings_weekday.py
산출
  outputs/ensemble/warnings_weekday.csv          — 평일 행만, warnings_daily.csv 와 같은 열
  outputs/ensemble/warnings_weekday_report.json  — 비율·기준·개발/시험 성적(전체 날짜·평일)
"""
import json

import numpy as np

from common import DEV, OUT, TEST
from warnings_model import counts, daily_table, evaluate, rolling_probs

WD = "월화수목금토일"


def main():
    D = daily_table("15")
    weekday = (D["off"] == 0).to_numpy()
    # rolling_probs 는 급락을 off == 1 인 날만 학습·예측함 → 복사본의 off 를 모두 1로 두어 전체 날짜를 쓰게 함
    # (off 는 급락 성분의 입력 변수가 아니라 고르는 데만 쓰이므로 확률 계산에는 영향 없음)
    A = D.copy()
    A["off"] = 1
    probs = rolling_probs(A, "crash", "crash")
    res, pe = evaluate(A, probs, "crash", "crash", final=True)

    y = D["crash"].to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(probs["logit"])
    span = lambda a, b: (D["date"] >= a).to_numpy() & (D["date"] < b).to_numpy()
    for name, (a, b) in [("dev", DEV), ("test", TEST)]:
        m = ok & weekday & span(a, b)
        res[f"{name}_weekday"] = {"days": int(m.sum()), "events": int(y[m].sum()),
                                  "alarm": counts(pe[m], y[m], res["th_alarm"]),
                                  "watch_or_alarm": counts(pe[m], y[m], res["th_watch"])}
    with open(OUT / "warnings_weekday_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)

    level = np.where(pe >= res["th_alarm"], "경보", np.where(pe >= res["th_watch"], "주의", "정상"))
    out = D.loc[weekday & ~np.isnan(probs["logit"]), ["date", "dow", "holiday_name", "crash", "mid_min", "day_max",
                                                       "max_hour", "ref28", "first3"]].copy()
    i = out.index
    out.insert(0, "판단시각", "15시")
    out.insert(1, "과제", "급락(평일)")
    out["요일"] = out["dow"].map(lambda d: WD[int(d)])
    out["확률"], out["단계"] = pe[i], level[i]
    out["p_로지스틱"], out["p_LightGBM"], out["p_가격신호"] = probs["logit"][i], probs["lgbm"][i], probs["price"][i]
    out = out.rename(columns={"holiday_name": "휴일명", "crash": "실제", "mid_min": "한낮최저", "day_max": "일최고",
                              "max_hour": "최고시각", "ref28": "직전28일기준", "first3": "월초1_3일"})
    cols = ["판단시각", "과제", "date", "요일", "휴일명", "확률", "단계", "실제", "한낮최저", "일최고", "최고시각",
            "직전28일기준", "월초1_3일", "p_로지스틱", "p_LightGBM", "p_가격신호"]
    out[cols].round(4).to_csv(OUT / "warnings_weekday.csv", index=False, encoding="utf-8-sig")

    d, t = res["dev"], res["test"]
    print(f"[15시 급락 · 전체 날짜] 가중 {res['weights']}  주의 ≥ {res['th_watch']:.4f} · 경보 ≥ {res['th_alarm']:.4f}")
    print(f"  개발 전체: {d['events']}/{d['days']}일 · 경보 {d['alarm']} · 주의 이상 {d['watch_or_alarm']}")
    print(f"  시험 전체: {t['events']}/{t['days']}일 · 경보 {t['alarm']} · 주의 이상 {t['watch_or_alarm']}")
    for name, nm in [("dev_weekday", "개발 평일"), ("test_weekday", "시험 평일")]:
        r = res[name]
        print(f"  {nm}: {r['events']}/{r['days']}일 · 경보 {r['alarm']} · 주의 이상 {r['watch_or_alarm']}")
    print(f"저장 warnings_weekday.csv : {out['date'].min()} ~ {out['date'].max()} ({len(out)}일)")


if __name__ == "__main__":
    main()
