# -*- coding: utf-8 -*-
"""
step3_merit_features.py — 급전순위 공개자료 기반 연료비 변수 생성 (v1.0)
==============================================================================
목적
  · 전력거래소 「변동비순위」 공개자료로부터 **월별 연료비 구조 변수**를 생성
  · 시간별 데이터셋에 결합하여 SMP 예측 모델의 입력변수로 사용

채택 근거 (2026.9.1 실증) — G-1 재판정 통과
  · 종전 내부 제원 기반 연료비 변수는 보유율 29.4%로 분포 불일치가 발생하여 기각
  · 공개자료 확보로 보유율 **78.0%**로 확대되어 재실증한 결과 **양 홀드아웃 구간 모두 개선**

    | 홀드아웃      | 기본 64종 | +연료비 4종 | 증감      |
    |--------------|----------|------------|----------|
    | 2026-01~08   | 8.92%    | **8.75%**  | −0.17%p  |
    | 2025-09~12   | 11.03%   | **10.83%** | −0.20%p  |

  · 종전 기각 사례(2분할 구조)와 달리 **부호가 일관**되어 재현성 확보
  · 지평별 : D+1 −0.17%p / D+3 −0.14%p / D+5 −0.13%p (6개 지평 평균 −0.08%p)
  · 한계비용~SMP 상관 : 내부 제원 0.554 → **공개자료 0.766**

채택 변수 4종 — 조합 탐색 결과 최적(합계 −0.37%p)
  | 변수            | 정의                                        |
  |-----------------|---------------------------------------------|
  | pm_mc_q60       | 급전순위 곡선의 누적용량 60% 지점 변동비      |
  | pm_vc_mean      | 중앙급전발전기 평균 변동비                    |
  | pm_vc_lng       | LNG 발전기 평균 변동비                        |
  | pm_mc_q60_chg   | pm_mc_q60 전월 대비 변화율                    |

  · 단독 투입(q60만)은 오히려 악화(+0.03%p) — 수준·구성·변화 3요소 병행이 필요
  · 전체 13종 대비 4종이 우수 — 과적합 억제 효과

운영 가용성
  · 변동비순위는 매월 **적용 전월에 사전 공표**되므로 예측 시점에 확보 가능
  · 신규 월 자료는 전력거래소 홈페이지에서 수집 후 `step7_parse_meritfile.py` 실행

산출
  · data/interim/merit_features_monthly.csv — ym × 변수
  · logs/step3_merit_features_{일시}.csv

실행
  python scripts\\step3_merit_features.py
  python scripts\\step3_merit_features.py --all      # 전체 13종 산출(분석용)

버전 : v1.0 (2026-09-01)
"""
import os
import sys
import argparse
import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERIM = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")

SRC = os.path.join(INTERIM, "src_merit_public.csv")
OUT = os.path.join(INTERIM, "merit_features_monthly.csv")

# 채택 변수 — 2026.9.1 조합 탐색 결과
CORE = ["pm_mc_q60", "pm_vc_mean", "pm_vc_lng", "pm_mc_q60_chg"]

_rows_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {값:<26} {비고}")
    _rows_log.append({"항목": 항목, "값": 값, "비고": 비고})


def build(m):
    """월별 급전순위 곡선 지표 산출

    비고 : 누적용량 분위수 지점의 변동비 = 해당 부하율에서의 한계발전기 변동비
    """
    rows = []
    for ym, s in m.groupby("ym"):
        s = s.sort_values("var_cost")
        s = s.assign(cum=s["cap_max"].cumsum())
        tot = float(s["cap_max"].sum())
        if tot <= 0:
            continue
        r = {"ym": ym, "pm_tot_cap": tot, "pm_n": len(s),
             "pm_vc_mean": float(s["var_cost"].mean()),
             "pm_vc_p75": float(s["var_cost"].quantile(0.75))}
        for q in (50, 60, 70, 80, 90):
            i = s["cum"].searchsorted(tot * q / 100)
            r[f"pm_mc_q{q}"] = float(s["var_cost"].iloc[min(i, len(s) - 1)])
        ft = s["fuel_type"].astype(str)
        lng = s.loc[ft.str.contains("LNG", na=False), "var_cost"]
        coal = s.loc[ft.str.contains("유연탄|무연탄|석탄", na=False), "var_cost"]
        r["pm_vc_lng"] = float(lng.mean()) if len(lng) else np.nan
        r["pm_vc_coal"] = float(coal.mean()) if len(coal) else np.nan
        rows.append(r)

    d = pd.DataFrame(rows).sort_values("ym").reset_index(drop=True)
    # 파생 — 변화율·이동평균·연료간 격차
    d["pm_mc_q60_chg"] = d["pm_mc_q60"].pct_change()
    d["pm_mc_q60_ma3"] = d["pm_mc_q60"].rolling(3, min_periods=1).mean()
    d["pm_spread"] = d["pm_vc_lng"] - d["pm_vc_coal"]
    return d


def save_log(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(LOG_DIR, f"{name}_{ts}.csv")
    pd.DataFrame(_rows_log).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def main():
    ap = argparse.ArgumentParser(description="급전순위 공개자료 연료비 변수 생성")
    ap.add_argument("--all", action="store_true",
                    help="전체 13종 산출(분석용). 미지정 시 채택 4종 + 식별자")
    args = ap.parse_args()

    print("=" * 74)
    print(" step3_merit_features — 연료비 구조 변수 생성 v1.0")
    print("=" * 74)

    if not os.path.exists(SRC):
        sys.exit(f"[중단] 입력파일 없음 → {SRC}\n"
                 f"       python scripts\\step7_parse_meritfile.py 선행 실행 필요")

    print("\n■ ① 원천 적재")
    m = pd.read_csv(SRC, encoding="utf-8-sig")
    n0 = len(m)
    m = m[(m["var_cost"].fillna(0) > 0) & (m["cap_max"].fillna(0) > 0)]
    rep("공개자료", f"{n0:,}행 → 유효 {len(m):,}행",
        f"{m['ym'].min()} ~ {m['ym'].max()} / {m['ym'].nunique()}개월")

    print("\n■ ② 변수 생성")
    d = build(m)
    rep("산출 변수", f"{len([c for c in d.columns if c.startswith('pm_')])}종",
        f"{len(d)}개월")
    rep("채택 변수", ", ".join(CORE), "2026.9.1 실증 최적 조합")

    print("\n■ ③ 변수 요약")
    print(f"\n  {'변수':<18}{'평균':>10}{'최소':>10}{'최대':>10}{'결측':>7}")
    for c in CORE:
        s = d[c]
        print(f"  {c:<18}{s.mean():>10.2f}{s.min():>10.2f}{s.max():>10.2f}"
              f"{s.isna().sum():>7}")

    print("\n■ ④ 산출")
    cols = ["ym"] + ([c for c in d.columns if c.startswith("pm_")] if args.all
                     else CORE)
    os.makedirs(INTERIM, exist_ok=True)
    d[cols].to_csv(OUT, index=False, encoding="utf-8-sig")
    rep("산출 파일", os.path.relpath(OUT, BASE_DIR),
        f"{len(d)}행 × {len(cols)}열")
    rep("결합 방법", "ym 기준 left join", "시간별 데이터셋에 월별 값 전개")

    p = save_log("step3_merit_features")
    print(f"\n[완료] 로그 → {os.path.relpath(p, BASE_DIR)}")
    print("\n다음 단계 : python scripts\\step3_features.py --merit-feat")


if __name__ == "__main__":
    main()
