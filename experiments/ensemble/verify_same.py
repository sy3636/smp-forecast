# -*- coding: utf-8 -*-
"""
verify_same.py — 내가 돌린 결과가 공유받은 기준 결과와 같은지 확인 (팀원 재현용)
==============================================================================
  · 비교 파일 : price_predictions.csv · warnings_daily.csv · explain_daily.csv
  · 비교 기간 : 기본 2026-01-01~10-06 (대시보드 기간)
  · 숫자 열은 값이, 글자 열(단계·근거 문장 등)은 내용이 같은지 행마다 비교

실행 (student-practice 최상위에서)
  python experiments/ensemble/verify_same.py --ref <공유받은 team_share 폴더>
"""
import argparse
from pathlib import Path

import pandas as pd

FILES = {"price_predictions.csv": ["date", "hour"],
         "warnings_daily.csv": ["판단시각", "과제", "date"],
         "explain_daily.csv": ["date"]}


def compare(ref, mine, keys):
    j = ref.merge(mine, on=keys, how="outer", suffixes=("_기준", "_내것"), indicator=True)
    only = int((j["_merge"] != "both").sum())
    diffs = {}
    for c in ref.columns:
        if c in keys or c + "_내것" not in j:
            continue
        a, b = j[c + "_기준"], j[c + "_내것"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = (a - b).abs()
            n = int(((d > 1e-9) | (a.isna() != b.isna())).sum())
            if n:
                diffs[c] = f"{n}행 다름(최대 {d.max():.4g})"
        else:
            n = int((a.fillna("").astype(str) != b.fillna("").astype(str)).sum())
            if n:
                diffs[c] = f"{n}행 다름"
    return only, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="공유받은 기준 결과 CSV 가 있는 폴더 (team_share)")
    ap.add_argument("--mine", default=str(Path(__file__).resolve().parents[2] / "outputs" / "ensemble"))
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-10-06")
    args = ap.parse_args()

    same = True
    for f, keys in FILES.items():
        rd = lambda d: pd.read_csv(Path(d) / f, encoding="utf-8-sig")
        ref, mine = rd(args.ref), rd(args.mine)
        ref, mine = (x[(x["date"] >= args.start) & (x["date"] <= args.end)] for x in (ref, mine))
        only, diffs = compare(ref, mine, keys)
        ok = not only and not diffs
        print(f"{f:<22} {len(ref):>5}행 · " + ("모두 같음" if ok else f"한쪽에만 있는 행 {only}개 · {diffs}"))
        same &= ok
    print(f"\n결과({args.start}~{args.end}): " + ("기준 결과와 완전히 같습니다 ✓" if same else "다른 곳이 있습니다 — 위 목록을 확인하세요 ✕"))


if __name__ == "__main__":
    main()
