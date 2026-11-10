# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step1 국제 연료가 지수 수집
================================================
목적 : 과거 구간(2021~2024) 발전기 열량단가 소급 환산에 사용할
       공개 국제 연료가 지수·환율을 확보(방식 C)

배경
  · 발전기 제원(열량단가 포함) 보유 구간이 2025-01 이후로 한정
  · 2025년 값을 그대로 소급 시 2022년 에너지 위기기에 102원 괴리 발생(2026.8.29 검증)
  · 물리 제원은 소급 가능하나 열량단가는 연료가 변동에 직결 → 공개 지수로 환산 필요

수집 항목 및 출처
  · 두바이유(POILDUBUSDM, $/bbl)         — IMF Primary Commodity Prices / FRED
  · 아시아 LNG(PNGASJPUSDM, $/MMBtu)     — IMF Primary Commodity Prices / FRED
  · 호주 유연탄(PCOALAUUSDM, $/ton)      — IMF Primary Commodity Prices / FRED
  · 원/달러 환율(EXKOUS, 원/USD 월평균)   — Board of Governors of the FRS / FRED
  · 기준 시점 : 2021-01 ~ 2026-06 (66개월), 수집일 2026-08-29

운영 방식
  · 원천 스냅샷 : data/external/fuel_index_fred_{수집일}.csv (수동 갱신 대비 보존)
  · 사내망에서 외부 접속이 차단된 경우 스냅샷만으로 재현 가능하도록 설계
  · --refresh 지정 시 FRED 재수집 시도(차단 시 스냅샷 사용)

산출 : data/interim/fuel_index_monthly.csv
       (ym, 원지수 4종, 원화환산 지수 3종 — idx_lng_krw·idx_coal_krw·idx_oil_krw)
       logs/step1_collect_fuel_index_{일시}.csv

유의 : 본 지수는 국제 시장가격이며 발전기별 개별 계약조건(직도입·개별요금제 등)은
       미반영. 열량단가 소급값은 근사치이며 원천 자료(자료요청서 v3.1 R-1) 대체 불가

버전 : v1.0 (2026-08-29)
실행 : python scripts\\step1_collect_fuel_index.py
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 미설치 → pip install -r requirements.txt")

EXTERNAL = os.path.join(BASE_DIR, "data", "external")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUT = os.path.join(INTERIM, "fuel_index_monthly.csv")

FRED = {
    "dubai_usd_bbl": ("POILDUBUSDM", "두바이유($/bbl)", "IMF Primary Commodity Prices"),
    "lng_asia_usd_mmbtu": ("PNGASJPUSDM", "아시아 LNG($/MMBtu)", "IMF Primary Commodity Prices"),
    "coal_au_usd_ton": ("PCOALAUUSDM", "호주 유연탄($/ton)", "IMF Primary Commodity Prices"),
    "krw_per_usd": ("EXKOUS", "원/달러 환율", "Board of Governors of the FRS"),
}

_rep = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<24} {값}   {비고}")
    _rep.append({"항목": 항목, "값": str(값), "비고": 비고})


def latest_snapshot():
    fs = sorted(glob.glob(os.path.join(EXTERNAL, "fuel_index_fred_*.csv")))
    return fs[-1] if fs else None


def main():
    ap = argparse.ArgumentParser(description="국제 연료가 지수 수집")
    ap.add_argument("--refresh", action="store_true",
                    help="FRED 재수집 시도(외부 접속 차단 시 스냅샷 사용)")
    args = ap.parse_args()

    print("=" * 70)
    print(" step1 국제 연료가 지수 수집 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    snap = latest_snapshot()
    if snap is None:
        sys.exit(f"[중단] 원천 스냅샷 없음 → {EXTERNAL}/fuel_index_fred_*.csv")

    print("\n■ ① 원천 적재")
    df = pd.read_csv(snap, encoding="utf-8-sig")
    rep("스냅샷", os.path.basename(snap), f"{len(df)}개월")
    rep("수집 구간", f"{df['ym'].min()} ~ {df['ym'].max()}")

    if args.refresh:
        try:
            import urllib.request as _u
            got = {}
            for col, (sid, _, _) in FRED.items():
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                with _u.urlopen(url, timeout=20) as r:
                    got[col] = pd.read_csv(r)
            rep("FRED 재수집", "성공", "스냅샷 갱신 권장")
        except Exception as e:                                   # noqa: BLE001
            rep("FRED 재수집", "실패 — 스냅샷 사용",
                f"{type(e).__name__} (사내망 외부접속 차단 추정)")

    print("\n■ ② 정합성 검증")
    miss = int(df.isna().sum().sum())
    rep("결측", f"{miss}건", "정상" if miss == 0 else "확인 필요")
    dup = int(df["ym"].duplicated().sum())
    rep("중복 연월", f"{dup}건", "정상" if dup == 0 else "확인 필요")
    for c in ("dubai_usd_bbl", "lng_asia_usd_mmbtu", "coal_au_usd_ton", "krw_per_usd"):
        rep(FRED[c][1], f"{df[c].min():.2f} ~ {df[c].max():.2f}",
            f"평균 {df[c].mean():.2f}")

    print("\n■ ③ 원화 환산 지수 산출")
    #  · 국제가격(USD) × 환율 → 원화 기준 지수. 열량단가 비율 산정에 사용
    df["idx_oil_krw"] = (df["dubai_usd_bbl"] * df["krw_per_usd"]).round(2)
    df["idx_lng_krw"] = (df["lng_asia_usd_mmbtu"] * df["krw_per_usd"]).round(2)
    df["idx_coal_krw"] = (df["coal_au_usd_ton"] * df["krw_per_usd"]).round(2)
    rep("환산 지수", "3종 산출", "idx_oil_krw · idx_lng_krw · idx_coal_krw")

    os.makedirs(INTERIM, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    rep("산출 파일", os.path.relpath(OUT, BASE_DIR))

    os.makedirs(LOG_DIR, exist_ok=True)
    lp = os.path.join(LOG_DIR, f"step1_collect_fuel_index_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(lp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["항목", "값", "비고"])
        w.writeheader()
        w.writerows(_rep)
    print(f"\n[완료] 로그 → {os.path.relpath(lp, BASE_DIR)}")
    print("\n다음 단계 : python scripts\\step3_fuel_features.py --backfill index")


if __name__ == "__main__":
    main()
