# -*- coding: utf-8 -*-
"""
step8_kpx_reference_check.py — KPX 수요예측 비교근거 대조 (v1.0)
==============================================================================
목적
  · 전력거래소(KPX) 수요예측 실무 기준 3종을 예외일 과대예측 진단·편의보정 실증의
    **대조 기준(비교근거)**으로 적용하고, 자체 데이터로 재현 여부를 검증
  · 기준값 : config/kpx_reference.json (출처 : 중급 전력거래반 교재 07번)

비교근거
  R1 (07번 10쪽) KSLF 일간 수요예측 — 가격결정은 모형예측값 사용, 오차율 기준 1.3%(2021)
      → 입력(수요예측) 결함 배제 단계의 기준선 : 일유형별 수요예측 오차 대리지표 산출
  R2 (07번 54·55쪽) 일유형별 알고리즘 매핑 — 월요일·징검다리·특수일·명절±2일 별도 취급
      → 편의보정 분류(현행 3분류 : 평일·토·일공휴)의 이질성 점검
  R3 (07번 9쪽) 72h 기온 1℃당 하계 4GW·동계 1GW, 태양광 이용률 1%p당 0.2GW
      → 자체 데이터 민감도 재현 + 수요→SMP 환산계수로 기대효과 정량 추정

산출
  logs/step8_kpx_reference_check_{일시}_R1.csv / _R2_oof.csv / _R2_ops.csv / _R3.csv
  표준출력 요약

실행
  python scripts/step8_kpx_reference_check.py
  python scripts/step8_kpx_reference_check.py --date 2026-10-05   (특정일 일유형·기온 기대변동 조회)

버전 : v1.0 (2026-10-04) 신설
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "processed", "smp_dataset_hourly.csv")
OOF = os.path.join(BASE, "data", "processed", "oof_predictions_smp_land.csv")
OPS = os.path.join(BASE, "outputs", "predict", "predictions_smp_land.csv")
REF = os.path.join(BASE, "config", "kpx_reference.json")
LOG = os.path.join(BASE, "logs")

WORK_CLASSES = ["화–금", "월요일", "연휴 익일", "연휴 전일", "징검다리 평일", "명절 전후"]


# ─────────────────────────── 일유형 분류(KPX 매핑 준용) ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kpx_daytype import build_calendar  # noqa: E402


def residual_table(df, ycol, pcol):
    df = df.assign(err=df[pcol] - df[ycol])
    base = df[df.kpx_class == "화–금"].err.mean()
    t = (df.groupby(["cls3", "kpx_class"])
         .agg(일수=("date", "nunique"), MAE=("err", lambda s: s.abs().mean()),
              편의=("err", "mean"), 평균실적=(ycol, "mean"))
         .reset_index())
    t["편의율(%)"] = t.편의 / t.평균실적 * 100
    t["화–금 대비 편의차"] = t.편의 - base
    return t.round(2).sort_values(["cls3", "일수"], ascending=[True, False])


# ─────────────────────────── R1 수요예측 오차 대리지표 ───────────────────────────
def r1_demand(d, cal):
    x = d[d.demand_actual.notna() & d.demand_fcst_land.notna()].copy()
    x["ratio"] = x.demand_actual / x.demand_fcst_land
    # 정의 차이(계통·시장 수요 범위) 제거 : 시간대별 직전 28일 비율 중위수로 기준화 — 대상일 미포함
    x = x.sort_values(["hour_n", "date"])
    x["ratio_ref"] = (x.groupby("hour_n").ratio
                      .transform(lambda s: s.shift(1).rolling(28, min_periods=14).median()))
    x = x[x.ratio_ref.notna()]
    x["fcst_adj"] = x.demand_fcst_land * x.ratio_ref
    x["ape"] = (x.fcst_adj - x.demand_actual).abs() / x.demand_actual * 100
    day = (x.groupby("date").apply(lambda g: pd.Series({
        "ape_h": g.ape.mean(),
        "ape_peak": abs(g.fcst_adj.max() - g.demand_actual.max()) / g.demand_actual.max() * 100}))
        .reset_index().merge(cal, on="date"))
    t = (day.groupby(["cls3", "kpx_class"])
         .agg(일수=("date", "size"), 시간별MAPE=("ape_h", "mean"), 일최대MAPE=("ape_peak", "mean"))
         .reset_index().round(2))
    tot = {"구간": f"{x.date.min()}~{x.date.max()}", "시간별MAPE": round(day.ape_h.mean(), 2),
           "일최대MAPE": round(day.ape_peak.mean(), 2),
           "정의차이_비율중위": round(x.ratio.median(), 3)}
    return t, tot


# ─────────────────────────── R3 민감도 재현 ───────────────────────────
def ols_slope(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    s2 = resid @ resid / max(1, len(y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    return b[1], np.sqrt(cov[1, 1])


def r3_sensitivity(d, cal):
    d = d.sort_values(["date", "hour_n"]).copy()
    d["t72"] = d.temp_land.rolling(72, min_periods=48).mean()
    day = (d.groupby("date").agg(peak_f=("demand_fcst_land", "max"), peak_a=("demand_actual", "max"),
                                 t72=("t72", "last"), smp=("smp_land", "mean"),
                                 dem=("demand_fcst_land", "mean"))
           .reset_index().merge(cal, on="date"))
    day["dt"] = pd.to_datetime(day.date)
    day["m"], day["y"] = day.dt.dt.month, day.dt.dt.year
    work = day[day.kpx_class.isin(["화–금"])].dropna(subset=["t72", "peak_f"])
    rows = []
    for season, months in (("하계(7~8월)", [7, 8]), ("동계(12~2월)", [12, 1, 2])):
        w = work[work.m.isin(months)]
        for col, lab in (("peak_f", "수요예측 일최대"), ("peak_a", "수요실적 일최대")):
            ww = w.dropna(subset=[col])
            if len(ww) < 30:
                continue
            # 연도 고정효과(수요 성장) + 요일 고정효과 통제
            X = pd.get_dummies(ww[["y", "dow"]].astype(str), drop_first=True).astype(float).values
            Xf = np.column_stack([ww.t72.values, X])
            b, se = ols_slope(ww[col].values / 1000, Xf)
            kpx = {"하계(7~8월)": 4.0, "동계(12~2월)": -1.0}[season]
            rows.append({"계절": season, "대상": lab, "표본일": len(ww),
                         "기울기(GW/℃)": round(b, 2), "표준오차": round(se, 2),
                         "KPX기준(GW/℃)": kpx, "기준대비": round(b / kpx, 2)})
    # 수요 → SMP 환산계수 : 연월 고정효과(연료가격 수준) 통제
    conv = []
    for season, months in (("하계(7~8월)", [7, 8]), ("동계(12~2월)", [12, 1, 2]), ("전체", list(range(1, 13)))):
        w = day[day.m.isin(months) & day.off.eq(False)].dropna(subset=["dem", "smp"])
        X = pd.get_dummies(w.dt.dt.strftime("%Y-%m"), drop_first=True).astype(float).values
        b, se = ols_slope(w.smp.values, np.column_stack([w.dem.values / 1000, X]))
        conv.append({"계절": season, "표본일": len(w), "SMP/수요(원/kWh per GW)": round(b, 2), "표준오차": round(se, 2)})
    return pd.DataFrame(rows), pd.DataFrame(conv), day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="특정일 일유형·기온 기대변동 조회(YYYY-MM-DD)")
    a = ap.parse_args()
    ref = json.load(open(REF, encoding="utf-8"))
    d = pd.read_csv(DS, encoding="utf-8-sig", low_memory=False)
    d["date"] = d.date.astype(str).str[:10]
    cal = build_calendar(d)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG, exist_ok=True)

    if a.date:
        r = cal[cal.date == a.date]
        print(r.to_string(index=False) if len(r) else f"[{a.date}] 데이터셋 범위 밖")
        return

    # R1
    t1, tot1 = r1_demand(d, cal)
    t1.to_csv(os.path.join(LOG, f"step8_kpx_reference_check_{ts}_R1.csv"), index=False, encoding="utf-8-sig")
    print("\n[R1] KPX 수요예측 오차 대리지표 (정의차이 기준화 후) — KPX 기준", ref["R1_kslf"]["수요예측오차율_기준"]["KPX"], "%")
    print(tot1); print(t1.to_string(index=False))

    # R2
    oof = pd.read_csv(OOF, encoding="utf-8-sig"); oof["date"] = oof.date.astype(str).str[:10]
    t2 = residual_table(oof.merge(cal, on="date"), "y", "pred")
    t2.to_csv(os.path.join(LOG, f"step8_kpx_reference_check_{ts}_R2_oof.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[R2] OOF 잔차(예측−실적) 일유형별 — {oof.date.min()}~{oof.date.max()}")
    print(t2.to_string(index=False))
    if os.path.exists(OPS):
        op = pd.read_csv(OPS, encoding="utf-8-sig")
        op = op.rename(columns={"예측": "pred", "실적": "y"}).dropna(subset=["y"])
        op["date"] = op.date.astype(str).str[:10]
        op = op.sort_values("예측생성시각").groupby(["date", "hour_n"]).tail(1)
        t2o = residual_table(op.merge(cal, on="date"), "y", "pred")
        t2o.to_csv(os.path.join(LOG, f"step8_kpx_reference_check_{ts}_R2_ops.csv"), index=False, encoding="utf-8-sig")
        print(f"\n[R2] 운영 예측 이력(보정 후) 일유형별 — {op.date.min()}~{op.date.max()}")
        print(t2o.to_string(index=False))

    # R3
    t3, conv, _ = r3_sensitivity(d, cal)
    pd.concat([t3, conv], axis=0).to_csv(os.path.join(LOG, f"step8_kpx_reference_check_{ts}_R3.csv"),
                                         index=False, encoding="utf-8-sig")
    print("\n[R3] 72h 기온 민감도 재현(화–금, 연도·요일 통제)"); print(t3.to_string(index=False))
    print("\n[R3] 수요→SMP 환산계수(근무일, 연월 고정효과)"); print(conv.to_string(index=False))


if __name__ == "__main__":
    main()
