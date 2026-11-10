# -*- coding: utf-8 -*-
"""
step1_parse_pss_demand.py — KPX 「하루전 발전계획용 수요예측」 공표자료 파싱 (v1.0)
==============================================================================
목적
  · 전력거래소 홈페이지 공표 「일자별 발전계획용 수요예측」 연도별 엑셀을 파싱
  · 육지·제주 2021~2026 시간별 수요예측(MW)을 단일 시계열로 통합

배경
  · 종전 수요예측은 공공데이터포털 API(계통한계가격 및 수요예측)로 수집하였으나
    **23시경 갱신**되어 예측 시각(14~15시)에는 D+1분이 미확보(2026.8.12 실측)
  · 이에 따라 수요예측 12종을 **운영 모델에서 제외**하고 있었음
  · 본 자료는 동일 정보를 **웹으로 공표**하는 경로이며, 갱신 시점이 더 이르면
    운영 투입이 가능해짐

기대 효과 (2026.9.1 실증)
  · 수요예측 12종 투입 시 **3개 홀드아웃 구간 전량 개선**

    | 홀드아웃      | 기준    | +수요예측 | 증감      |
    |--------------|--------|----------|----------|
    | 2026-01~08   | 9.46%  | 8.77%    | −0.69%p  |
    | 2025-09~12   | 11.27% | 10.81%   | −0.46%p  |
    | 2025-01~08   | 7.97%  | 7.59%    | −0.38%p  |

  · 평균 **−0.51%p** — 검증한 변수군 중 최대 효과
    (기상 −0.40%p, 연료비 −0.17%p 대비)

자료 구조
  · 1행 : 표제(일자별 발전계획용 수요예측)
  · 2행 : 헤더 — 구분 · 1h ~ 24h
  · 3행~: 일자(YYYYMMDD) + 시간별 수요예측(MW)
  · 파일명 : pssInLand_{연도}.xlsx (육지) / pssJeju_{연도}.xlsx (제주)

산출
  · data/interim/pss_demand_hourly.csv
    컬럼 : date, hour_n, pss_demand_land, pss_demand_jeju
  · logs/step1_parse_pss_demand_{일시}.csv

실행
  python scripts\\step1_parse_pss_demand.py
  python scripts\\step1_parse_pss_demand.py --src "경로"

출처 : 전력거래소 「하루전 발전계획용 수요예측」
       육지 https://new.kpx.or.kr/pssInland.es?mid=a10606090100
       제주 https://new.kpx.or.kr/pssJeju.es?mid=a10606090200

버전 : v1.0 (2026-09-01)
"""
import os
import re
import sys
import glob
import argparse
import datetime

import pandas as pd

try:
    import openpyxl
except ImportError:
    sys.exit("[중단] openpyxl 미설치 — pip install openpyxl")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM = os.path.join(BASE_DIR, "data", "interim")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUT = os.path.join(INTERIM, "pss_demand_hourly.csv")

DEFAULT_SRC = os.path.join(RAW_DIR, "supplement", "pss_demand")

_rows_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<26} {값:<26} {비고}")
    _rows_log.append({"항목": 항목, "값": 값, "비고": 비고})


def parse_file(path):
    """단일 연도 파일 파싱 — 일자 × 24시간 → 장형(long) 변환

    비고 : read_only 모드는 본 파일의 차원(max_row·max_column)을 1로 반환하여
           셀 접근이 불가하므로 **일반 모드 + iter_rows 순차 접근**을 사용함
           (2026.8.31 step7_fuel_price.py 동일 사례)
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]

    hdr_idx, hcol = None, {}
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if not row:
            continue
        c0 = str(row[0] or "").strip()

        # 헤더 행 — '구분' + '1h'~'24h'
        if hdr_idx is None:
            if c0 == "구분":
                hdr_idx = i
                for j, v in enumerate(row):
                    m = re.match(r"^(\d{1,2})h$", str(v or "").strip().lower())
                    if m:
                        hcol[int(m.group(1))] = j
            continue

        s = c0.replace("-", "").replace(".", "")
        if not re.match(r"^\d{8}$", s):
            continue
        date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        for h, j in hcol.items():
            if j < len(row) and isinstance(row[j], (int, float)):
                rows.append({"date": date, "hour_n": h, "value": float(row[j])})
    wb.close()

    if hdr_idx is None:
        return pd.DataFrame(), "헤더 미탐색"
    if len(hcol) < 24:
        return pd.DataFrame(), f"시간 열 부족({len(hcol)}/24)"
    return pd.DataFrame(rows), None


def region_of(fname):
    """파일명에서 계통 구분 판정"""
    f = fname.lower()
    if "inland" in f:
        return "land"
    if "jeju" in f:
        return "jeju"
    return None


def save_log(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(LOG_DIR, f"{name}_{ts}.csv")
    pd.DataFrame(_rows_log).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def main():
    ap = argparse.ArgumentParser(description="KPX 하루전 발전계획용 수요예측 파싱")
    ap.add_argument("--src", default=DEFAULT_SRC, help="엑셀 보관 폴더")
    args = ap.parse_args()

    print("=" * 74)
    print(" step1_parse_pss_demand — 하루전 발전계획용 수요예측 파싱 v1.0")
    print("=" * 74)

    if not os.path.isdir(args.src):
        sys.exit(f"[중단] 폴더 없음 → {args.src}")

    files = sorted(glob.glob(os.path.join(args.src, "pss*.xlsx")))
    print("\n■ ① 대상 파일")
    rep("보관 폴더", os.path.relpath(args.src, BASE_DIR))
    rep("엑셀 파일", f"{len(files)}개")

    print("\n■ ② 파싱")
    parts = {"land": [], "jeju": []}
    fails = []
    for p in files:
        fn = os.path.basename(p)
        reg = region_of(fn)
        if reg is None:
            fails.append((fn, "계통 구분 미판독"))
            continue
        df, err = parse_file(p)
        if err:
            fails.append((fn, err))
            continue
        parts[reg].append(df)
        rep(f"  {fn[:26]}", f"{len(df):,}행",
            f"{df['date'].min()} ~ {df['date'].max()}")

    if not parts["land"]:
        sys.exit("[중단] 육지 파싱 결과 없음")
    if fails:
        for fn, e in fails:
            rep("  실패", fn[:30], e)

    print("\n■ ③ 통합")
    out = None
    for reg, col in (("land", "pss_demand_land"), ("jeju", "pss_demand_jeju")):
        if not parts[reg]:
            rep(f"{reg} 자료", "없음 — 생략")
            continue
        d = pd.concat(parts[reg], ignore_index=True)
        n0 = len(d)
        d = d.drop_duplicates(subset=["date", "hour_n"], keep="last")
        d = d.rename(columns={"value": col})
        rep(f"{reg} 통합", f"{n0:,} → {len(d):,}행",
            f"{d['date'].min()} ~ {d['date'].max()}")
        out = d if out is None else out.merge(d, on=["date", "hour_n"], how="outer")

    out = out.sort_values(["date", "hour_n"]).reset_index(drop=True)

    print("\n■ ④ 정합성 검증")
    rep("통합 결과", f"{len(out):,}행",
        f"{out['date'].min()} ~ {out['date'].max()}")
    nd = out["date"].nunique()
    rep("일수 · 시간", f"{nd:,}일 × 24시간", f"기대 {nd * 24:,}행")
    for c in [c for c in out.columns if c.startswith("pss_")]:
        s = out[c]
        rep(f"{c}", f"보유 {s.notna().mean() * 100:.1f}%",
            f"평균 {s.mean():,.0f}MW / 범위 {s.min():,.0f}~{s.max():,.0f}")
    # 시간 결측일 확인
    cnt = out.groupby("date").size()
    bad = cnt[cnt != 24]
    rep("시간 불완전 일자", f"{len(bad)}일",
        ", ".join(bad.index[:3]) if len(bad) else "없음")

    print("\n■ ⑤ 산출")
    os.makedirs(INTERIM, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    rep("산출 파일", os.path.relpath(OUT, BASE_DIR), f"{len(out):,}행")

    p = save_log("step1_parse_pss_demand")
    print(f"\n[완료] 로그 → {os.path.relpath(p, BASE_DIR)}")


if __name__ == "__main__":
    main()
