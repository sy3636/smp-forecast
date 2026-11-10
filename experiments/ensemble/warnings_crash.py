# -*- coding: utf-8 -*-
"""
warnings_crash.py — 육지 급락 경보 v2 「한낮 최저가 예측 중앙값 앙상블」 (2026-11-08, 팀 작업본)
==============================================================================
방식 : 다음 날 육지 11~15시 최저가(원)를 모델 14개로 예측 → 14개의 **중앙값** → 낮을수록 위험한 점수
       → 매달 1일 그 이전 자료로 로지스틱 보정해 확률로 바꿈 → 「경보」 한 단계, 휴일·주말에만
  · 모델 14개 (매달 1일, 그 이전 자료로만 재학습 — crash_reg_ens.py 와 같은 설정)
      LightGBM(중앙값·Huber) · CatBoost(중앙값·RMSE) · XGBoost(절대오차) · HistGB(절대오차) · RandomForest · ExtraTrees
      · Ridge · Huber · KNN · SVR · MLP · Chronos-2 15시 한낮 최저 예측
  · 변수 : 15시에 알 수 있는 것만 — 자체 수요 추정·원전 뺀 수요·태양광 비중·Chronos 15시 예측·어제/지난주 가격·연휴 구조·계절 (max_search.rich)
고른 과정 (학습 구간 2022–2024만, crash_reg_ens.py) : 후보 29개(단독 14 + 앙상블 15) 중
  적중일 ≥ 교체 전 −1(8일) 안에서 헛경보 최소 → 「예측 중앙값 앙상블」(학습 헛경보 36일, AP 0.375; LightGBM 단독 49일)
  평일은 학습 구간 급락 0일이라 경보 안 냄. 휴일 기준 = 학습 구간 휴일 확률 분위(5~99.5%) 중 같은 규칙으로 고름.
시험(2025-01-01~2026-10-31) : 적중 29/39 · 헛경보 60일 (v1 가격신호+로지스틱: 32/39 · 107일, 처음 모델: 39/39 · 414일)
판단 시각 : 전날 15시.  실행 : python experiments/ensemble/warnings_crash.py (~5분; events.py 다음, 다시 events.py)
산출 : outputs/ensemble/warnings_crash.csv · warnings_crash_report.json   (v1 은 _backup/warnings_crash_v1_*)
"""
import json
import warnings

import numpy as np
import pandas as pd

import chronos_mix as CM
import crash_reg_ens as CRE
import max_search as MS
import reg_search as RS
import tune_alerts as TA
import warn_search as W
from common import OUT, SETTINGS, rd

warnings.filterwarnings("ignore")
WD = "월화수목금토일"
QS = [round(q, 3) for q in list(np.arange(0.05, 0.99, 0.02)) + [0.99, 0.995]]
BASELINE = CRE.BASELINE          # 교체 전(처음) 모델의 학습 구간 성적 — 고정값 (경보 적중 8 · 주의 이상 9)


def main():
    daily = rd(OUT / "events_daily.csv")
    D = MS.rich(W.build("crash", daily), "crash")
    dates, y = D["date"].to_numpy(), D["y"].to_numpy()
    t = RS.target("crash", D)
    R = CRE.rolling(D, MS.rich_cols(D, "crash"), t)
    R["chronos_mid"] = D["pr_mid_min"].to_numpy(float)
    # crash_reg_ens.py 에서 고른 그대로: 중앙값(가능한 모델끼리, 2021년은 Chronos 만)을 전체 기간으로 보정,
    # 학습 구간·경보 날짜는 LightGBM 확률이 있는 날(표본 외 학습이 시작된 뒤)로 한정
    med = np.nanmedian(np.vstack(list(R.values())), 0)
    p = CM.calibrate(pd.DataFrame({"date": dates, "s": -med}), y)
    ref = CM.calibrate(pd.DataFrame({"date": dates, "s": -R["lgb_q50"]}), y)
    off = D["off"].to_numpy() == 1
    dev = (dates >= TA.DEV[0]) & (dates <= TA.DEV[1]) & ~np.isnan(y) & ~np.isnan(ref)
    p = np.where(np.isnan(p), np.nanmedian(p[dev]), p)
    p = np.where(np.isnan(ref), np.nan, p)
    need = max(BASELINE["경보적중"], BASELINE["주의이상적중"]) - max(1, round(0.05 * BASELINE["사건일"]))   # 8일
    m = dev & off
    best = None
    for q in QS:
        th = float(np.quantile(p[m], q))
        hit, fa = int(((p >= th) & m & (y == 1)).sum()), int(((p >= th) & m & (y == 0)).sum())
        if hit >= need and (best is None or (fa, -hit) < (best[2], -best[1])):
            best = (th, hit, fa, q)
    th, hit, fa, q = best
    level = np.where(np.isnan(p), "", np.where(off & (p >= th), "경보", "정상"))
    hour = SETTINGS["issue_hour"]["crash"]
    out = pd.DataFrame({"판단시각": f"{hour}시", "과제": "육지 급락", "date": dates, "요일": [WD[int(d)] for d in D["dow"]],
                        "휴일명": D["holiday_name"], "구분": np.where(off, "휴일", "평일"), "확률": p, "단계": level, "실제": y,
                        "한낮최저": D["mid_min"], "예측_한낮최저_중앙값": med, **{f"예측_{k}": v for k, v in R.items()}})
    out = out[out["단계"] != ""]
    out.round(4).to_csv(OUT / "warnings_crash.csv", index=False, encoding="utf-8-sig")
    rep = {"model": "한낮 최저가 예측 중앙값 앙상블(14개)", "members": list(R), "threshold_holiday": th, "threshold_quantile": q,
           "weekday": "경보 안 냄", "levels": "경보 한 단계", "dev": {"휴일_적중일": hit, "휴일_헛경보일": fa, "사건일": int(y[dev].sum())},
           "dev_baseline": BASELINE, "issue_hour": hour}
    (OUT / "warnings_crash_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[육지 급락 v2 · 전날 {hour}시] {out['date'].min()} ~ {out['date'].max()} · 휴일 경보 ≥ {th:.4f} (학습 휴일 확률 {q} 분위) · 평일 경보 없음")
    print(f"  학습: 적중 {hit}/{int(y[dev].sum())}일 · 헛경보 {fa}일")


if __name__ == "__main__":
    main()
