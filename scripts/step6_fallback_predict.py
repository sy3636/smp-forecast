# -*- coding: utf-8 -*-
"""
step6_fallback_predict.py — 시간별 예측 폴백 산출 (v1.0)
==============================================================================
배경 (A-9, 2026. 9. 8. 사건)
  · 2026. 9. 9. 대상 시간별 예측이 **18:00·21:00·23:30 3단 전량 실패**
  · 원인 : D+1 수요예측 11/12종 미공표 — 23:30 초과(종전 실측 경계 14~20시)
  · 구조적 취약점 : 3단은 실행 시각만 다른 **동일 모델**이므로 진정한 폴백이 아님

목적
  · 수요예측 미사용 **47변수 축약 모델**(주간 예측)의 D+1 결과를
    시간별 예측 산출물 형식으로 변환하여 **산출 실패 0건** 확보
  · 정확도 열위(2026. 9. 3. 실측 0.06원 · 0.7%)를 수용하고 **가용성 우선**

원칙
  · 산출물에 **폴백 산출 사실을 명기** — 정확도 차이 투명 고지
  · 편의 보정은 운영과 동일 적용(창 21일 · γ1.5 · 평일·휴일 2분류)
  · 정규 산출물이 이미 존재하면 **덮어쓰지 않음**(--force 로만 강제)

제주 경로 (v1.1, 2026.9.10 신설)
  · 주간 예측은 육지 전용이므로, 제주는 **육지 폴백값 + 최근 격차**로 산출
      제주 = 육지폴백(보정 후) + mean_최근K일(제주실적 − 육지실적) [동일 시간대]
  · 근거 : 제주 2단계 모델의 spread_ma7 논리와 동일. 실적만 사용하므로
           수요예측 미공표 상황에서도 산출 가능
  · 한계 : 2단계 잔차모델을 사용하지 않는 **근사치** — 산출구분 열에 명기

실행
  python scripts\\step6_fallback_predict.py --date 20260909
  python scripts\\step6_fallback_predict.py --date 20260909 --base 20260908
  python scripts\\step6_fallback_predict.py --date 20260909 --no-bias

제주 mirror (v1.2, 2026.10.1)
  · JEJU_MODE=mirror(기본) : 제주 = 육지 폴백값(격차 0) — 정규 산출의 제주 mirror 채택과 일치
  · JEJU_MODE=stage2 : 종전 v1.1 근사(육지 폴백 + 최근 14일 격차)

버전 : v1.2 (2026-10-01) — 제주 mirror 연동
       v1.1 (2026-09-10) — 제주 근사 경로 신설
"""
import os
import sys
import glob
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
#  산출 구분 표기 — 주석 행이 아닌 **열**로 기록
#   · 2026.9.9 결함 조치 : 주석 행을 넣으면 시간별 예측 로더가 이를 헤더로
#     인식하여 열 구성이 깨지고 대시보드가 오류(st.columns(0))를 일으킴
#   · 월별 전망만 주석을 지원하므로, 시간별 산출물은 열로 표기함
JEJU_MIRROR = (os.environ.get("JEJU_MODE", "mirror") or "mirror") == "mirror"
KIND_NOTE = ("폴백(수요예측 미사용 47변수 축약 모델) — "
             "D+1 수요예측 미공표로 정규 모델 산출 불가. "
             + ("제주는 육지 폴백값 적용(mirror)" if JEJU_MIRROR
                else "제주는 육지 폴백값 + 최근 격차 근사(2단계 잔차모델 미적용)"))
SPREAD_DAYS = 14      # 제주·육지 격차 산정 창(일) — 2026.9.10 신설
_rows = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<22} {str(값):<26} {비고}", flush=True)
    _rows.append({"항목": 항목, "값": str(값), "비고": 비고})


def latest_weekly(base=None):
    """주간 예측 산출물 경로 — base 미지정 시 최신"""
    if base:
        p = os.path.join(OUT_DIR, f"SMP주간예측_{base}.csv")
        if not os.path.exists(p):
            sys.exit(f"[중단] 주간 예측 없음 → {p}")
        return p
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "SMP주간예측_*.csv")))
    if not fs:
        sys.exit("[중단] 주간 예측 산출물 없음 — step6_weekly_predict.py 선행")
    return fs[-1]


def main():
    ap = argparse.ArgumentParser(description="시간별 예측 폴백 산출")
    ap.add_argument("--date", help="대상일 YYYYMMDD(기본 익일)")
    ap.add_argument("--base", help="주간 예측 기준일 YYYYMMDD(기본 최신)")
    ap.add_argument("--no-bias", action="store_true", help="편의 보정 생략")
    ap.add_argument("--force", action="store_true",
                    help="정규 산출물이 있어도 덮어씀")
    a = ap.parse_args()
    tgt = a.date or (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    tgt_dash = f"{tgt[:4]}-{tgt[4:6]}-{tgt[6:8]}"

    print("=" * 78)
    print(" step6_fallback_predict — 시간별 예측 폴백 산출 v1.2")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 대상일 : {tgt_dash}")
    print("=" * 78)

    dst = os.path.join(OUT_DIR, f"SMP예측_{tgt}.csv")
    if os.path.exists(dst) and not a.force:
        sys.exit(f"[중단] 정규 산출물 존재 → {os.path.relpath(dst, BASE_DIR)}\n"
                 "       덮어쓰려면 --force 지정")

    wp = latest_weekly(a.base)
    w = pd.read_csv(wp, encoding="utf-8-sig")
    w.columns = [c.lstrip("﻿") for c in w.columns]
    w["대상일"] = w["대상일"].astype(str).str[:10]
    seg = w[w["대상일"] == tgt_dash].sort_values("거래시간")
    print("\n■ ① 원천")
    rep("주간 예측", os.path.basename(wp), f"기준일 {w['기준일'].iloc[0]}")
    if seg.empty:
        sys.exit(f"[중단] 주간 예측에 {tgt_dash} 없음 — 지평 범위 초과")
    rep("대상 지평", seg["지평"].iloc[0], f"{len(seg)}시간")
    if len(seg) != 24:
        rep("주의", f"{len(seg)}시간만 확보", "24시간 미만 — 산출 계속")

    out = pd.DataFrame({
        "date": tgt_dash,
        "hour_n": seg["거래시간"].astype(int).values,
        "예측_smp_land": np.round(seg["예측"].astype(float).values, 2),
        "실적_smp_land": np.nan,
    })

    #  편의 보정 — 운영과 동일 조건
    print("\n■ ② 편의 보정")
    if a.no_bias:
        rep("보정", "생략", "--no-bias 지정")
        out["예측_smp_land_보정전"] = out["예측_smp_land"]
    else:
        try:
            from step6_bias_adjust import apply_bias, GAMMA, CAP
            raw = out["예측_smp_land"].values.copy()
            p, note = apply_bias(raw.copy(), out["hour_n"].values,
                                 "smp_land", tgt_dash)
            out["예측_smp_land_보정전"] = np.round(raw, 2)
            out["예측_smp_land"] = np.round(p, 2)
            adj = out["예측_smp_land"] - out["예측_smp_land_보정전"]
            rep("보정 적용", note, f"γ={GAMMA} / 상한 ±{CAP:.0f}원")
            rep("보정폭", f"평균 {adj.mean():+.1f}원",
                f"범위 {adj.min():+.1f} ~ {adj.max():+.1f}")
        except Exception as e:
            rep("보정", "실패", f"{type(e).__name__} — 무보정 산출")
            out["예측_smp_land_보정전"] = out["예측_smp_land"]

    #  제주 — mirror(v1.2 기본) : 육지 폴백값 / stage2 : 육지 폴백값 + 최근 격차(v1.1)
    out["예측_smp_jeju"] = np.nan
    out["실적_smp_jeju"] = np.nan
    if JEJU_MIRROR:
        out["예측_smp_jeju"] = out["예측_smp_land"].values
        rep("제주 계통", "mirror — 육지 폴백값 적용",
            f"일평균 {out['예측_smp_jeju'].mean():.1f}원")
    else:
        try:
            ds = os.path.join(BASE_DIR, "data", "processed", "smp_dataset_hourly.csv")
            d = pd.read_csv(ds, encoding="utf-8-sig", low_memory=False)
            d.columns = [c.lstrip("\ufeff") for c in d.columns]
            d["date"] = d["date"].astype(str).str[:10]
            pr = d[(d["date"] < tgt_dash)].dropna(subset=["smp_land", "smp_jeju"])
            sel = sorted(pr["date"].unique())[-SPREAD_DAYS:]
            pr = pr[pr["date"].isin(sel)]
            if pr["date"].nunique() >= 3:
                sp = (pr["smp_jeju"] - pr["smp_land"]).groupby(pr["hour_n"]).mean()
                gap = out["hour_n"].map(sp).fillna(0.0).values
                out["예측_smp_jeju"] = np.round(out["예측_smp_land"].values + gap, 2)
                rep("제주 계통", f"근사 산출 {len(sel)}일 격차",
                    f"평균 격차 {gap.mean():+.1f}원 / "
                    f"일평균 {out['예측_smp_jeju'].mean():.1f}원")
            else:
                rep("제주 계통", "미산출", f"격차 산정 이력 부족({pr['date'].nunique()}일)")
        except Exception as e:
            rep("제주 계통", "미산출", f"{type(e).__name__}")

    out["산출구분"] = KIND_NOTE
    print("\n■ ③ 산출")
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    rep("산출물", os.path.relpath(dst, BASE_DIR),
        "산출구분 열에 폴백 명기")
    rep("육지 일평균", f"{out['예측_smp_land'].mean():.1f}원/kWh",
        f"최고 {out['예측_smp_land'].max():.1f}"
        f"({int(out.loc[out['예측_smp_land'].idxmax(), 'hour_n'])}시) / "
        f"최저 {out['예측_smp_land'].min():.1f}"
        f"({int(out.loc[out['예측_smp_land'].idxmin(), 'hour_n'])}시)")

    #  예측 이력 반영 — 폴백 구분 표기
    hist = os.path.join(OUT_DIR, "predictions_smp_land.csv")
    rec = pd.DataFrame({
        "예측생성시각": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "date": out["date"], "hour_n": out["hour_n"],
        "예측": out["예측_smp_land"], "실적": np.nan})
    if os.path.exists(hist):
        old = pd.read_csv(hist, encoding="utf-8-sig")
        old = old[old["date"].astype(str) != tgt_dash]
        rec = pd.concat([old, rec], ignore_index=True)
    rec.to_csv(hist, index=False, encoding="utf-8-sig")
    if out["예측_smp_jeju"].notna().any():
        hj = os.path.join(OUT_DIR, "predictions_smp_jeju.csv")
        rj = pd.DataFrame({
            "예측생성시각": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            "date": out["date"], "hour_n": out["hour_n"],
            "예측": out["예측_smp_jeju"], "실적": np.nan})
        if os.path.exists(hj):
            oj = pd.read_csv(hj, encoding="utf-8-sig")
            oj = oj[oj["date"].astype(str) != tgt_dash]
            rj = pd.concat([oj, rj], ignore_index=True)
        rj.to_csv(hj, index=False, encoding="utf-8-sig")
    rep("예측 이력", "반영", "육지·제주 — 동일 대상일 기존 행 대체")

    os.makedirs(LOG_DIR, exist_ok=True)
    lp = os.path.join(LOG_DIR,
                      f"step6_fallback_{tgt}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    pd.DataFrame(_rows).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 폴백 산출 — 로그 {os.path.relpath(lp, BASE_DIR)}")
    print("  ※ 정규 모델 산출이 가능해지면 step6_predict.py 재실행으로 갱신할 것")
    print("=" * 78)


if __name__ == "__main__":
    main()
