# -*- coding: utf-8 -*-
"""check_csv.py — 팀원용 검산: 결과 CSV 폴더의 2026-01-01~10-06 숫자를 TEAM_GUIDE.md 7-5 검산표 형식으로 출력

실행  python check_csv.py [결과 CSV 폴더]     # 폴더를 빼면 현재 폴더
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A, B = "2026-01-01", "2026-10-06"                       # 대시보드 기간
DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
rd = lambda f: pd.read_csv(DIR / f, encoding="utf-8-sig")
period = lambda d: d[(d["date"] >= A) & (d["date"] <= B)]

# 1) 가격 지표 — 실적·예측·전일동시간이 모두 있는 시간
p = period(rd("price_predictions.csv"))
q = p.dropna(subset=["실적", "예측", "전일동시간"])
y, f, n1 = q["실적"], q["예측"], q["전일동시간"]
den = np.maximum(y, 20)
mae = np.abs(f - y).mean()

# 2) 경보 — (과제, 판단시각, 기준) → 실제·적중·헛경보·놓침·판단 일수
w = period(rd("warnings_daily.csv")).dropna(subset=["실제"])
cnt = {}
for (task, when), g in w.groupby(["과제", "판단시각"]):
    real = g["실제"] == 1
    for name, on in [("경보만", g["단계"] == "경보"), ("주의 이상", g["단계"].isin(["주의", "경보"]))]:
        cnt[task, when, name] = (real.sum(), (on & real).sum(), (on & ~real).sum(), (~on & real).sum(), len(g))

# 대표 숫자 — 발표·대시보드 카드에 쓰는 세 줄
c, s = cnt["급락", "15시", "주의 이상"], cnt["급등+25%", "날짜 단위", "경보만"]
print(f"[대표 숫자] {A}~{B}")
print(f"1. 급락경보 : {c[1]}/{c[0]}일 적중 (15시 기준·주의 이상, 헛경보 {c[2]}일)")
print(f"2. 가격예측 : {mae:.2f}원 (평균 오차 MAE)")
print(f"3. 급등예측 : {s[1]}/{s[0]}일 적중 (경보 기준, 헛경보 {s[2]}일)")

print("\n[자세한 검산표]")
print(f"MAE {mae:.2f}원 · rMAE {mae / np.abs(n1 - y).mean():.4f} · "
      f"MAPE {(np.abs(f - y) / den).mean() * 100:.2f}% · 편의 {((f - y) / den).mean() * 100:+.2f}%")

# 80% 범위 — 실적·하한·상한이 있는 시간
r = p.dropna(subset=["실적", "하한80", "상한80"])
y, lo, hi = r["실적"], r["하한80"], r["상한80"]
print(f"포함률 {((y >= lo) & (y <= hi)).mean() * 100:.1f}% · 평균 폭 {(hi - lo).mean():.1f}원 · "
      f"하한 아래 {(y < lo).mean() * 100:.1f}% · 상한 위 {(y > hi).mean() * 100:.1f}%")

for (task, when, name), (n, tp, fp, fn, days) in cnt.items():
    print(f"{task} {when} {name}: 실제 {n}일 · 적중 {tp} · 헛경보 {fp} · 놓침 {fn} ({days}일 판단)")

# 과제 3 — 기준 + 7개 항목 = 그날 평균 예측
e = period(rd("explain_daily.csv"))
parts = ["최근 가격", "전력 수요", "태양광", "요일·휴일", "연료비·원전", "계절·시간", "다른 모델 보정"]
print(f"근거 {len(e)}일 · 검산 최대 오차 {(e['ref'] + e[parts].sum(axis=1) - e['ens']).abs().max():.2f}원")
print("영향 큰 순:", e[parts[:6]].abs().mean().sort_values(ascending=False).round(1).to_dict())
