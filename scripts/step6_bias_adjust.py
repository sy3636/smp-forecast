# -*- coding: utf-8 -*-
"""
step6_bias_adjust.py — 시간대별 잔차 편의 보정 (v1.0)
==============================================================================
목적
  · 직전 K일 **운영 예측 이력**의 시간대별 평균 잔차를 익일 예측값에 가산하여
    계통적 과소·과대예측을 상쇄

배경 (2026.9.2 진단)
  · 9.1~9.2 산출에서 주간(10~13시)·저녁(19~22시) 최대 30원 과소예측 발생
  · 원인 분석 결과 2건 확인
    ① 수준 편의 : 일평균 수준 ↔ 편의 상관 **−0.896**
       → 수준이 높은 날 과소, 낮은 날 과대 (트리 모델의 평균회귀 축소)
    ② 진폭 축소 : 일중 진폭이 실적의 **88.8%**

실증 결과 (step8_bias_test.py, 2026.9.2)
  | 보정 | 전체 MAPE | 급등구간(8월~) | 판정 |
  |------|----------|---------------|------|
  | 무보정 | 7.52% | 7.88% | 기준 |
  | **잔차 보정(K=21, γ=0.7)** | **7.31%** | **6.75%** | **채택** |
  | 선형 재보정 | 15.22% | 18.71% | 기각 — 과보정 발산 |
  | 진폭 확대 | 7.50% | 7.10% | 기각 — 개선 미미 |
  | 결합 | 7.68% | 7.32% | 기각 — 전체 악화 |

  · 육지 −0.21%p(급등 −1.13%p), 제주 −0.73%p(급등 −1.80%p)
  · K·γ 격자 탐색에서 K=14~28 구간 편차 0.03%p 이내 — 파라미터 민감도 낮음

설계 원칙
  · 보정계수는 **대상일 직전까지 확보된 실적**만 사용 → 데이터 누출 없음
  · 감쇠계수 γ 적용 — 일시적 이상치에 대한 과보정 억제
  · 보정폭 상한 ±25원/kWh — 이력 이상 시 산출 왜곡 방지
  · 이력 부족(< 7일) 시 **보정 미적용**하고 사유를 반환

산식
  ŷ'(h) = ŷ(h) + γ · mean_{최근 K일}( 실적(h) − 예측(h) )

실행
  본 모듈은 step6_predict.py 에서 호출되며 단독 실행은 점검 용도
  python scripts\\step6_bias_adjust.py                (육지 현재 계수 확인)
  python scripts\\step6_bias_adjust.py --target smp_jeju

감쇠계수 개정 (step8_regime_test.py, 2026.9.5) — γ 0.7 → 1.2
  · 배경 : 9월 수준 급락 국면에서 편의 +9.69% 확대, 전환 대응력 부족 확인
  · 격자 탐색(K 7~28 × γ 0.5~2.0, 육지 17일·408행)
    - γ 방향 단조 개선(0.5<0.7<1.0<1.2<1.5), γ 2.0에서 반전 — 내부 최적 확인
    - K 영향 미미 : γ 1.5 기준 K 7~21 편차 0.31원 → **창 21일 존치**
  · 시간 분할 검증(전반 8일 탐색 → 후반 9일 적용)
    - 후반 MAE 15.35 → 14.38원(−0.97) — **표본 외 개선 확인, 과적합 아님**
  · 제주 계통 동일 방향 확인(K21 기준 γ0.7 10.80 → γ1.2 9.80원)
  · 채택값 : γ **1.2** — 격자 최적 1.5보다 보수적 설정
    (소표본 17일, γ>1은 잔차 과반영 구간이므로 단계적 상향)
  · 기각 : 전환 감지 기반 적응형 보정(전환 시 K5·γ1.0) — 단순 γ 상향 대비 열위

  | 구성(육지, K21) | 전체MAE | 전환MAE | 평시MAE |
  |------|---------|---------|---------|
  | 무보정 | 12.77 | 19.92 | 11.24 |
  | γ0.7(종전) | 11.41 | 18.13 | 9.97 |
  | **γ1.2(채택)** | **10.76** | **16.99** | **9.43** |
  | γ1.5(격자 최적) | 10.59 | 16.48 | 9.32 |

요일 구분 분리 (step8_daytype_test.py, 2026.9.6) — 평일·휴일 2분류 적용
  · 배경 : 휴일(토·일·공휴일)은 수요 감소로 SMP 수준 하락
    - 학습구간(2021.1.1~2026.9.4, 2,072일) 일평균 : 평일 140.16 / 휴일 124.31원(-11.3%)
  · 진단 : 운영 이력 24일 오차 분해 — **휴일 성능이 평일의 2~3배 열위**
    | 구분 | 일수 | MAE | 편의 |
    |------|-----:|----:|-----:|
    | 평일 | 18 | 7.85 | +0.10% |
    | 일·공휴일 | 4 | 14.32 | +13.33% |
    | 토 | 2 | 23.32 | +15.72% |
  · 원인 : 종전 보정은 시간대(24)별 잔차만 사용하여 창 21일 내 평일·휴일이 혼재
           → 휴일 예측에 평일 잔차가 섞여 보정 왜곡
  · 실증 결과(창 21일·γ1.2, 평가 17일·408행)
    | 구성 | 전체MAE | 평일 | 토 | 일공휴 |
    |------|--------:|-----:|----:|-------:|
    | 무보정 | 13.42 | 11.16 | 24.64 | 14.97 |
    | 시간대만(종전) | 10.54 | 7.32 | 22.76 | 15.27 |
    | **2분류(채택)** | **9.51** | **6.46** | **19.61** | **14.99** |
    | 3분류(평일·토·일공휴) | 9.93 | 6.46 | 22.76 | 15.27 |
    | 2분류 혼합 | 9.84 | 6.76 | 21.16 | 14.60 |
  · 채택 : **평일·휴일 2분류** — 전체 -1.03원(-9.8%), 평일·토·일공휴 전 구간 개선
  · 2026. 9. 27. 재실증 후 **3분류(평일·토·일공휴) 운영 채택** (부장 판정)
    — 토 이력 4일(8.29·9.05·9.12·9.19). 9.26은 연휴라 일공휴
    — 같은 날 재실증 전체 MAE는 현행 10.27 vs 3분류 11.29(+1.02). 채택 기준
      (휴일 개선+평일 비악화)는 미충족이나 판정 권한에 따라 채택
  · 구분별 이력 부족(< MIN_DAYS_SPLIT) 시 **전체 잔차로 자동 대체**

감쇠계수 재개정 (step8_hourgamma_test.py, 2026.9.7) — γ 1.2 → 1.5
  · 계기 : 2026.9.7 주간(10~16시) 편의 +10.60% 관측 — 시간대별 차등 검토
  · 실증(운영 이력 25일·평가 15일, 창 21일·2분류·상한 ±25원)
    | 구성 | 전체MAE | 주간MAE | 야간MAE | 전체편의 |
    |------|--------:|--------:|--------:|---------:|
    | 무보정 | 12.81 | 14.61 | 12.07 | 1.16% |
    | γ1.2(종전) | 8.58 | 10.80 | 7.66 | 0.86% |
    | 주간1.5·야간1.2 | 8.44 | 10.32 | 7.66 | 1.18% |
    | 주간1.8·야간1.2 | 8.43 | 10.28 | 7.66 | 1.50% |
    | 주간2.5·야간1.2 | 8.93 | 11.99 | 7.66 | 2.23% |
    | **균일 1.5(채택)** | **8.27** | **10.32** | **7.43** | **0.78%** |
  · 판정 : **시간대별 차등 기각** — 균일 상향이 전 구간 우월
    주간 특화는 야간을 개선하지 못하여 대조군(균일 1.5) 대비 열위
  · 채택 : γ **1.5** — 전체 −0.31원 · 주간 −0.48원 · 야간 −0.24원 동시 개선
  · 근거 보강 : 2026.9.5 격자 탐색에서도 γ1.5가 최적(K14 기준 10.49원),
    당시 소표본을 이유로 1.2로 보수 채택하였으나 이력 확대로 재판정

버전 : v1.8 (2026-10-04) — KPX 특수근무일(명절전후·징검다리) 평일 창 제외(K1)
       v1.7 (2026-09-27) — 3분류(평일·토·일공휴) 잔차 창 채택
       v1.6 (2026-09-21) — 이중창 혼합 최종 판정(평일 유지·휴일 완화 유지)
       v1.5 (2026-09-12) — 휴일 이중창 일시 완화
       v1.4 (2026-09-10) — 이중창 혼합 채택
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 운영 파라미터 (2026.9.5 개정 — 수준 전환 대응 격자 탐색) ──────────────
WINDOW = 21          # 잔차 산정 창(일) — K 민감도 낮음, 이력 안정성 우선
GAMMA = 1.5          # 감쇠계수 — 2026.9.5 0.7→1.2, 2026.9.7 1.2→1.5 상향
MIN_DAYS = 7         # 최소 이력 일수 — 미만이면 보정 미적용
CAP = 25.0           # 보정폭 상한(원/kWh)
SPLIT_DAYTYPE = True  # 일유형 구분 잔차 — 2026.9.6 2분류, 2026.9.27 3분류
SPLIT_3CLASS = True   # 평일·토·일공휴 — 2026.9.27 채택
MIN_DAYS_SPLIT = 4    # 구분별 최소 이력 일수 — 미만 시 전체 잔차로 대체

# ── 이중창 혼합 (2026.9.10 신설 — 하향 전환 국면 대응) ──────────────────
#  배경 : 2026.9.7~9.9 하향 전환 3일 연속(수준변화 −20.0 / −22.8 / −12.9%)
#         편의 +15.40% / MAE 19.02원까지 확대 — 창 21일 단독은 급변 추종 지체
#  실증 : 운영 이력 21일(2026-08-20~09-10) 기준
#    | 구성 | 전체MAE | 전환MAE | 평시MAE |
#    |------|--------:|--------:|--------:|
#    | 현행 K21 단독 | 12.07 | 17.72 | 9.81 |
#    | **이중창 혼합(K21·K5 평균)** | **11.62** | **16.66** | **9.61** |
#    | 짧은창 K7 단독 | 11.66 | 16.79 | 9.61 |
#    | 적응형(전환 감지) | 12.19 | 18.14 | 9.81 — 기각 |
#  · 격자 최적(K10·γ1.8) 11.63원과 사실상 동등하나 **γ 유지**로 진동 위험 회피
#  · 시간 분할 검증 : 후반(검증) 구간에서도 개선 — 일반화 확인
BLEND_DUAL = True     # 이중창 혼합 사용 — 2026.9.10 채택, 2026.9.21 최종 유지
WINDOW_SHORT = 5      # 단기 창(일)
BLEND_W = 0.5         # 단기 창 가중(0.5 = 균등 평균)
MIN_DAYS_SHORT = 3    # 단기 창 최소 이력 일수

# ── 휴일 이중창 일시 완화 (2026.9.12 신설) ──────────────────────────────
#  배경 : 2026.9.13(일) 보정 전 MAE 3.21원·편의 +0.02 → 보정 후 12.85원·−12.85
#         휴일 경로가 직전 주말 과대 잔차를 이중창·γ1.5로 과하게 하향 반영
#  대조(운영 이력 역산, 육지)
#    | 구성 | 9.12 MAE | 9.13 MAE |
#    |------|--------:|--------:|
#    | 현행 이중 γ1.5 ±25 | 10.04 | 12.85 |
#    | **휴일 단일 γ0.7 ±15** | **7.01** | **6.24** |
#  · 평일은 이중창·γ1.5 유지(9.11 과대 완화에 유효)
#  · 2026.9.21 최종 판정: 휴일 완화 유지
#  · 2026.9.27 3분류 채택 — 잔차 창만 평일·토·일공휴로 분리. 완화 파라미터는
#    토·일·공휴일(daytype_of='휴일')에 종전처럼 적용
HOLIDAY_RELAX = True  # 2026.9.12 적용, 2026.9.21 최종 유지 (주말 과소 재발 없음)
HOLIDAY_BLEND_DUAL = False   # 휴일: 단일창
HOLIDAY_GAMMA = 0.7          # 휴일: 감쇠 하향
HOLIDAY_CAP = 15.0           # 휴일: 상한 축소

# ── KPX 특수근무일 제외(비교군 K1 → 2026.10.4 운영 반영) ───────────────
#  · 근거 : 중급 07번 54·55쪽 + KPX비교근거_채택서 — 명절전후·징검다리가
#           「평일」 잔차 창에 혼재하여 과대보정(+4~5% 편의)
#  · 동작 : 평일 창 산정 시 해당일 이력을 제외(대상일이 특수근무일이어도 동일)
BIAS_K1_EXCLUDE = True


def _holiday_set():
    """공휴일 일자 집합(YYYY-MM-DD) — 원천 부재 시 빈 집합"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from step3_features import load_holiday_calendar
        return {d.strftime("%Y-%m-%d") for d in load_holiday_calendar()}
    except Exception:
        return set()


_HOL = None


def daytype_of(day):
    """일자 → '평일' | '휴일' (토·일·공휴일) — 완화 파라미터용"""
    global _HOL
    if _HOL is None:
        _HOL = _holiday_set()
    d = pd.to_datetime(str(day))
    return "휴일" if (d.weekday() >= 5 or str(day)[:10] in _HOL) else "평일"


def daytype3_of(day):
    """일자 → '평일' | '토' | '일공휴' — 잔차 창 분리용 (2026.9.27)"""
    global _HOL
    if _HOL is None:
        _HOL = _holiday_set()
    d = pd.to_datetime(str(day))
    if str(day)[:10] in _HOL or d.weekday() == 6:
        return "일공휴"
    if d.weekday() == 5:
        return "토"
    return "평일"


def load_history(t):
    """운영 예측 이력 적재 — 실적이 확보된 건만 반환

    비고 : 이력에는 운영 파이프라인의 전 후처리(기상 패턴 보정 등)가
           이미 반영되어 있으므로, 잔여 편의를 그대로 포착함
    """
    p = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    h = pd.read_csv(p, encoding="utf-8-sig")
    need = {"date", "hour_n", "예측", "실적"}
    if not need.issubset(h.columns):
        return pd.DataFrame()
    h = h.dropna(subset=["실적", "예측"])
    # 동일 대상일 중복 시 최신 산출분 채택
    if "예측생성시각" in h.columns:
        h = h.sort_values("예측생성시각").drop_duplicates(
            subset=["date", "hour_n"], keep="last")
    return h[["date", "hour_n", "예측", "실적"]]


def _special_weekday_dates(dates):
    """이력 일자 중 KPX 특수근무일(명절전후·징검다리) 집합."""
    global _HOL
    if not BIAS_K1_EXCLUDE or len(dates) == 0:
        return set()
    if _HOL is None:
        _HOL = _holiday_set()
    try:
        from kpx_daytype import EXCLUDE_FROM_WEEKDAY, build_calendar
        from step3_features import load_holiday_calendar_named
    except Exception:
        return set()
    hol = load_holiday_calendar_named()
    name_map = {}
    if hol is not None and len(hol):
        name_map = {pd.Timestamp(r.dt).strftime("%Y-%m-%d"): r.date_name
                    for r in hol.itertuples()}
    rows = []
    for d in dates:
        ds = str(d)[:10]
        dt = pd.to_datetime(ds)
        is_hol = (dt.weekday() >= 5 or ds in _HOL or ds in name_map)
        rows.append({"date": ds,
                     "daytype": "휴일" if is_hol else "평일",
                     "holiday_name": name_map.get(ds)})
    if not rows:
        return set()
    cal = build_calendar(pd.DataFrame(rows))
    return set(cal.loc[cal["kpx_class"].isin(EXCLUDE_FROM_WEEKDAY), "date"])


def hourly_bias(t, target_date, window=WINDOW, min_days=MIN_DAYS,
                split=SPLIT_DAYTYPE):
    """대상일 직전 window일의 시간대별 평균 잔차(실적−예측) 산출

    · split=True 이면 대상일과 동일 구분 이력만 사용
      — SPLIT_3CLASS 이면 평일·토·일공휴, 아니면 평일·휴일
      — 구분별 이력이 MIN_DAYS_SPLIT 미만이면 전체 잔차로 대체
    · BIAS_K1_EXCLUDE : 평일 창에서 명절전후·징검다리 제외

    반환 : (Series[hour_n → 보정값], 사유 문자열)
           보정 불가 시 (None, 사유)
    """
    h = load_history(t)
    if h.empty:
        return None, "예측 이력 없음"
    # 대상일 이전 실적만 사용 — 누출 차단
    h = h[h["date"] < str(target_date)]
    days = sorted(h["date"].unique())
    if len(days) < min_days:
        return None, f"이력 부족({len(days)}일 < {min_days}일)"

    if split:
        key = daytype3_of if SPLIT_3CLASS else daytype_of
        dt = key(target_date)
        h2 = h[h["date"].map(key) == dt]
        if dt == "평일" and BIAS_K1_EXCLUDE:
            bad = _special_weekday_dates(sorted(h2["date"].unique()))
            if bad:
                h2 = h2[~h2["date"].isin(bad)]
        d2 = sorted(h2["date"].unique())
        if len(d2) >= MIN_DAYS_SPLIT:
            sel = d2[-window:]
            hh = h2[h2["date"].isin(sel)]
            b = (hh["실적"] - hh["예측"]).groupby(hh["hour_n"]).mean()
            tag = "K1제외 " if (dt == "평일" and BIAS_K1_EXCLUDE) else ""
            return b, f"{tag}{dt} {len(sel)}일({sel[0]}~{sel[-1]})"
        #  구분별 이력 부족 — 전체 잔차로 대체
        sel = days[-window:]
        hh = h[h["date"].isin(sel)]
        b = (hh["실적"] - hh["예측"]).groupby(hh["hour_n"]).mean()
        return b, (f"전체 {len(sel)}일({sel[0]}~{sel[-1]}) "
                   f"— {dt} 이력 {len(d2)}일로 구분 미적용")

    sel = days[-window:]
    hh = h[h["date"].isin(sel)]
    b = (hh["실적"] - hh["예측"]).groupby(hh["hour_n"]).mean()
    return b, f"최근 {len(sel)}일({sel[0]}~{sel[-1]})"


def blended_bias(t, target_date, window=WINDOW, short=WINDOW_SHORT,
                 w=BLEND_W):
    """이중창 혼합 잔차 — 장기(K21)·단기(K5) 가중 평균

    · 단기 창 이력이 MIN_DAYS_SHORT 미만이면 장기 단독으로 대체
    · 일유형 분류(2·3분류)는 각 창에 동일하게 적용(hourly_bias 내부 처리)
    """
    b_long, why_long = hourly_bias(t, target_date, window)
    if b_long is None:
        return None, why_long
    b_short, why_short = hourly_bias(t, target_date, short,
                                     min_days=MIN_DAYS_SHORT)
    if b_short is None:
        return b_long, f"{why_long} / 단기창 미적용({why_short})"
    idx = b_long.index.union(b_short.index)
    bl = b_long.reindex(idx).fillna(0.0)
    bs = b_short.reindex(idx).fillna(0.0)
    return ((1.0 - w) * bl + w * bs), f"장기 {why_long} + 단기 {why_short}"


def _params_for(target_date, gamma=None, cap=None):
    """대상일 구분에 따른 보정 파라미터 — 휴일이면 완화값"""
    dt = daytype_of(target_date)
    use_dual = BLEND_DUAL
    g = GAMMA if gamma is None else gamma
    c = CAP if cap is None else cap
    tag = ""
    if HOLIDAY_RELAX and dt == "휴일":
        use_dual = HOLIDAY_BLEND_DUAL
        g = HOLIDAY_GAMMA if gamma is None else gamma
        c = HOLIDAY_CAP if cap is None else cap
        tag = "휴일완화"
    return dt, use_dual, g, c, tag


def apply_bias(p, hour_n, t, target_date,
               window=WINDOW, gamma=None, cap=None):
    """예측값에 시간대별 잔차 보정 적용

    인자
      p           : 예측값 배열
      hour_n      : 시간대 배열(1~24)
      t           : 타깃명(smp_land · smp_jeju)
      target_date : 예측 대상일 YYYY-MM-DD

    반환 : (보정 예측값, 비고 문자열)
    """
    dt, use_dual, g, c, tag = _params_for(target_date, gamma, cap)
    if use_dual:
        b, why = blended_bias(t, target_date, window)
        mode = f"이중창 K{window}·K{WINDOW_SHORT}(w={BLEND_W})"
    else:
        b, why = hourly_bias(t, target_date, window)
        mode = f"단일창 K{window}"
    if tag:
        mode = f"{mode}/{tag}"
    if b is None:
        return p, f"편의보정 미적용({why})"
    adj = pd.Series(hour_n).map(b).fillna(0.0).to_numpy(float)
    adj = np.clip(g * adj, -c, c)
    return p + adj, (f"편의보정 γ={g} / {mode} / {why} / "
                     f"보정 {adj.mean():+.1f}원(범위 {adj.min():+.1f}~{adj.max():+.1f})")


def main():
    import argparse
    import datetime
    ap = argparse.ArgumentParser(description="편의 보정계수 점검")
    ap.add_argument("--target", default="smp_land",
                    choices=["smp_land", "smp_jeju"])
    ap.add_argument("--date", default=None, help="대상일 YYYY-MM-DD(기본 익일)")
    args = ap.parse_args()
    d = args.date or (datetime.date.today() +
                      datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 74)
    print(" step6_bias_adjust — 편의 보정계수 점검 v1.6")
    print(f" 타깃 : {args.target} / 대상일 : {d}")
    print("=" * 74)

    dt, use_dual, g, c, tag = _params_for(d)
    if use_dual:
        b, why = blended_bias(args.target, d)
        mode = f"이중창 K{WINDOW}·K{WINDOW_SHORT}"
    else:
        b, why = hourly_bias(args.target, d)
        mode = f"단일창 K{WINDOW}"
    if b is None:
        print(f"\n  보정 미적용 — {why}")
        sys.exit(0)
    print(f"\n■ 산정 근거 : {why}")
    print(f"■ 파라미터   : {mode}"
          f"{('/' + tag) if tag else ''} / 감쇠 {g} / 상한 ±{c:.0f}원 "
          f"/ 구분 {dt}")
    print(f"\n  {'시간':>4}{'평균잔차':>10}{'적용보정':>10}")
    print("  " + "-" * 24)
    for h in range(1, 25):
        v = float(b.get(h, 0.0))
        a = float(np.clip(g * v, -c, c))
        print(f"  {h:>4}{v:>10.2f}{a:>10.2f}")
    print("  " + "-" * 24)
    print(f"  {'평균':>4}{b.mean():>10.2f}"
          f"{np.clip(g * b.mean(), -c, c):>10.2f}")


if __name__ == "__main__":
    main()
