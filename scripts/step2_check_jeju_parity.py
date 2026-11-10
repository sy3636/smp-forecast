# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - 제주·육지 SMP 동일성 감시
================================================
배경 : 제주는 HVDC 해저연계선을 통해 육지와 동일한 SMP를 가짐이 확인됨
       (사내 확인, 2026.8.11)
       → 2026.7 이후 일치율 100.0% — 제주 별도 예측 모델의 존치 실익 소멸

목적 : 동일성 전제가 계속 성립하는지 **정기 감시**하여, 붕괴 시 즉시 인지
       계통 운영 여건 변화(연계선 정지·제주 자체 급전 등)로 전제가 깨질 수 있음

판정 기준 (v1.1 개정 — 2026.9.7)
  · 정상   : 일치율 99.0% 이상 → 육지 예측값 그대로 적용 유지
  · 주의   : 95.0~99.0%, 또는 95.0% 미만이나 **평균절대차 ≤ 2.0원**
  · 재검토 : 95.0% 미만 **그리고 평균절대차 > 2.0원** → 제주 2단계 모델 재가동 검토
  · 유보   : 해당 월 표본 **15일(360시간) 미만** → 판정 제외(종합 판정에 미반영)

개정 사유
  · 종전은 일치율 단독 기준이어서, 소수점 차이가 소수 시간대에만 발생해도 등급 하락
  · 2026.9.7 사례 : 9월 일치율 92.86%로 「재검토」 판정되었으나
    - 표본 168시간(7일)에 불과, 평균절대차 0.402원(SMP 150원대의 0.27%)
    - 6월 99.86% · 7월 100.00% · 8월 98.12%로 최근 추세는 수렴
  · 월초 소표본이 매월 반복적으로 오탐을 유발하는 구조를 보완

입력 : data/processed/smp_dataset_hourly.csv
출력 : logs/jeju_parity_{일시}.csv  (월별 일치율·평균절대차)
       logs/jeju_parity_history.csv (누적 이력)

실행 : python scripts\\step2_check_jeju_parity.py
       python scripts\\step2_check_jeju_parity.py --months 24

버전 : v1.0 (2026-08-12)
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 미설치 → pip install -r requirements.txt")

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HIST = os.path.join(LOG_DIR, "jeju_parity_history.csv")

TOL = 0.005          # 동일 판정 허용오차(원/kWh) — 소수점 반올림 대응
OK, WARN = 99.0, 95.0
DIFF_TH = 2.0        # 평균절대차 임계(원/kWh) — 일치율 미달 시 실질 괴리 판단
MIN_HOURS = 360      # 판정 유효 최소 표본(시간) — 15일 미만은 유보


def main():
    ap = argparse.ArgumentParser(description="제주·육지 SMP 동일성 감시")
    ap.add_argument("--months", type=int, default=18, help="표시 개월 수(기본 18)")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now()

    print("=" * 74)
    print(" 제주·육지 SMP 동일성 감시 (v1.0)")
    print(f" 실행시각 : {now:%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    src = os.path.join(PROCESSED, "smp_dataset_hourly.csv")
    if not os.path.exists(src):
        sys.exit(f"[중단] 입력파일 없음 → {src}")

    d = pd.read_csv(src, encoding="utf-8-sig", low_memory=False,
                    usecols=["date", "hour_n", "smp_land", "smp_jeju"])
    d = d.dropna(subset=["smp_land", "smp_jeju"])
    if d.empty:
        sys.exit("[중단] 육지·제주 동시 확보 구간 없음")

    d["ym"] = d["date"].str[:7]
    gap = (d["smp_land"] - d["smp_jeju"]).abs()
    d["일치"] = gap < TOL
    d["절대차"] = gap

    g = d.groupby("ym").agg(
        시간수=("일치", "size"),
        일치율=("일치", lambda s: round(s.mean() * 100, 2)),
        평균절대차=("절대차", lambda s: round(s.mean(), 3)),
        최대절대차=("절대차", lambda s: round(s.max(), 2)),
    ).reset_index()

    def verdict(rate, diff, hours):
        """일치율·평균절대차·표본을 병용한 등급 판정

        · 표본 부족(15일 미만)은 '유보' — 종합 판정에서 제외
        · 일치율 미달이어도 평균절대차가 임계 이하면 '주의'로 완화
        """
        if hours < MIN_HOURS:
            return "유보"
        if rate >= OK:
            return "정상"
        if rate >= WARN:
            return "주의"
        return "재검토" if diff > DIFF_TH else "주의"

    g["판정"] = [verdict(r["일치율"], r["평균절대차"], r["시간수"])
                 for _, r in g.iterrows()]

    print(f"\n■ 월별 일치율 (최근 {args.months}개월)")
    print(f"  {'기준월':<10}{'시간수':>8}{'일치율':>9}{'평균절대차':>11}"
          f"{'최대절대차':>11}   판정")
    print("  " + "-" * 62)
    for _, r in g.tail(args.months).iterrows():
        print(f"  {r['ym']:<10}{r['시간수']:>8,}{r['일치율']:>8.2f}%"
              f"{r['평균절대차']:>11.3f}{r['최대절대차']:>11.2f}   {r['판정']}")
    print("  " + "-" * 62)

    # ── 종합 판정 : 최근 3개월 기준(유보 월 제외) ──
    recent = g.tail(3)
    valid = recent[recent["판정"] != "유보"]
    held = recent[recent["판정"] == "유보"]
    if valid.empty:
        valid = recent                      # 전량 유보 시 원본 사용(참고 판정)
    i = valid["일치율"].idxmin()
    worst = float(valid.loc[i, "일치율"])
    state = verdict(float(valid.loc[i, "일치율"]),
                    float(valid.loc[i, "평균절대차"]),
                    int(valid.loc[i, "시간수"]))

    print(f"\n■ 종합 판정 (최근 3개월 최저 {worst:.2f}% / "
          f"평균절대차 {float(valid.loc[i, '평균절대차']):.3f}원)")
    if not held.empty:
        print(f"  · 표본 부족 유보 : {' · '.join(held['ym'])} "
              f"(15일 미만 — 판정 제외)")
    if state == "정상":
        print("  [○] 정상 — 육지 예측값 직접 적용 유지 (제주 별도 모델 불요)")
    elif state == "주의":
        print("  [△] 주의 — 원인 확인 후 다음 달 재점검")
        print("       확인 : HVDC 연계선 정지·제주 자체 급전 여부(관련 부서 문의)")
    else:
        print("  [×] 재검토 — **제주 2단계 모델 재가동 검토 필요**")
        print("       절차 : python scripts\\step4_jeju_stage2.py 실행 후 성능 확인")

    # ── 이력 누적 ─────────────────────────────
    row = {
        "실행시각": f"{now:%Y-%m-%d %H:%M:%S}",
        "대상구간": f"{g['ym'].iloc[0]} ~ {g['ym'].iloc[-1]}",
        "최근3개월_최저일치율": worst,
        "판정": state,
        "최종월": g["ym"].iloc[-1],
        "최종월_일치율": g["일치율"].iloc[-1],
    }
    new = not os.path.exists(HIST)
    pd.DataFrame([row]).to_csv(HIST, mode="a", index=False,
                               header=new, encoding="utf-8-sig")

    out = os.path.join(LOG_DIR, f"jeju_parity_{now:%Y%m%d_%H%M%S}.csv")
    g.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  → 월별 상세 {out}")
    print(f"  → 누적 이력 {HIST}")
    print("=" * 74)


if __name__ == "__main__":
    main()
