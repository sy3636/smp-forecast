# -*- coding: utf-8 -*-
"""
step3_cross_hour.py — 시간 간 교차 시차(cross-hour lag) 생성 (v1.0)
==============================================================================
목적
  · 전일(D-1) **24시간 전량**을 각 행의 입력 변수로 부착
  · 동시간 시차(smp_lag_d1)만으로는 포착되지 않는 **일중 형상(shape) 정보** 확보

근거
  · Raviv, Bouwman & van Dijk(2013) — 시간별 가격 간 교차관계를 명시적으로
    모형화하면 일평균 예측 정확도가 15~20% 개선됨을 실증
  · 본 프로그램 실증(2026.9.2) : 단독 적용 시 MAE 8.72 → 8.59원(−1.5%)
    단, 편의가 +1.44 → +2.20%로 악화되어 **앙상블 1개 성분으로만 채택**

시점 정합성
  · 사용 자료는 **전일 실적** 뿐이므로 D+1 예측 시점(23:30)에 전량 확보 가능
  · 주간 예측(15시, D+2 이상)은 전일 실적이 미확보되므로 **적용 대상 아님**

생성 변수
  · d1_h1 ~ d1_h24 (24종) — 전일 각 시간대 SMP 실적

사용
  from step3_cross_hour import add_cross_hour, CROSS_COLS
  df, cols = add_cross_hour(df, "smp_land")

버전 : v1.0 (2026-09-02)
"""
import pandas as pd

PREFIX = "d1_h"
CROSS_COLS = [f"{PREFIX}{h}" for h in range(1, 25)]


def add_cross_hour(df, target, hist=None):
    """전일 24시간 실적을 24개 변수로 부착

    인자
      df     : date · hour_n · {target} 컬럼을 가진 데이터프레임
      target : 타깃명(smp_land · smp_jeju)
      hist   : 실적 원천을 별도 지정할 경우(예측 시 과거 실적표). 미지정 시 df 사용

    반환 : (변수 부착된 df, 생성 컬럼 목록)

    비고 : 원본에 동일 컬럼이 있으면 **덮어쓰기**하여 중복 부착을 방지함
    """
    src = df if hist is None else hist
    if target not in src.columns:
        return df, []

    piv = src.pivot_table(index="date", columns="hour_n",
                          values=target, aggfunc="first")
    piv = piv.reindex(columns=range(1, 25))
    piv.columns = CROSS_COLS
    piv = piv.sort_index().shift(1)          # 전일로 이동

    out = df.drop(columns=[c for c in CROSS_COLS if c in df.columns],
                  errors="ignore")
    out = out.merge(piv.reset_index(), on="date", how="left")
    return out, CROSS_COLS


def coverage(df):
    """부착 결과 점검 — 결측률 반환"""
    cols = [c for c in CROSS_COLS if c in df.columns]
    if not cols:
        return 0.0, 0
    return float(df[cols].notna().mean().mean()), len(cols)
