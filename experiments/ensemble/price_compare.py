# -*- coding: utf-8 -*-
"""
price_compare.py — 가격 앙상블 비교: 팀 원래(2021~ 학습) vs 2020~ 학습 LightGBM (2026-11-09, 팀 작업본) — 결과만 남김
==============================================================================
후보 (ensemble.py 와 같은 결합·보정·범위 방식, 성분만 바꿈)
  A 현재            : lgb_stack_ft_weekly · lgb_stack_weekly · lgb_l1_weekly · chr_ft_lora
  B 2020 학습        : 위 LightGBM 3개를 _y2020 판으로 (run_lgbm.py --train-from 2020-01-08 --tag _y2020 --stack-allow-missing)
  C 둘 다            : A 의 LightGBM 3 + B 의 LightGBM 3 + chr_ft_lora (가중이 매달 고름)
  D 2020 + 제로샷     : B + chr_dem_off
고르기 (학습 구간 2022–2024 만) : 가격 목표 5개(MAE ≤ 7.0 · rMAE ≤ 0.65 · MAPE < 5% · |편의| ≤ 1.5% · 범위 75~85%) 달성 수 → 목표까지 남은 거리 합
시험 구간(2025-01-01~2026-10-31)은 고른 것과 현재만 계산.
실행 : python experiments/ensemble/price_compare.py   산출 : outputs/ensemble/price_compare_report.json
"""
import copy
import json

import numpy as np

import ensemble as EN
from common import DEV, OUT, TEST

BASE = EN.CONFIG
L3 = ["lgb_stack_ft_weekly", "lgb_stack_weekly", "lgb_l1_weekly"]
CANDS = {
    "A 현재": L3 + ["chr_ft_lora"],
    "B 2020 학습": [m + "_y2020" for m in L3] + ["chr_ft_lora"],
    "C 둘 다": L3 + [m + "_y2020" for m in L3] + ["chr_ft_lora"],
    "D 2020 + 제로샷": [m + "_y2020" for m in L3] + ["chr_ft_lora", "chr_dem_off"],
}
TARGETS = [("MAE", lambda f, iv: f["MAE"], 7.0, "≤"), ("rMAE", lambda f, iv: f["rMAE"], 0.65, "≤"),
           ("MAPE", lambda f, iv: f["MAPE"], 5.0, "<"), ("편의", lambda f, iv: abs(f["bias"]), 1.5, "≤"),
           ("범위", lambda f, iv: iv["cover"], (75.0, 85.0), "in")]


def score(r):
    f, iv = r["final"], r["interval"]
    met, gap, vals = 0, 0.0, {}
    for name, get, tgt, op in TARGETS:
        v = get(f, iv)
        vals[name] = round(float(v), 4)
        if op == "in":
            ok = tgt[0] <= v <= tgt[1]
            g = 0 if ok else min(abs(v - tgt[0]), abs(v - tgt[1])) / 10
        else:
            ok = v < tgt if op == "<" else v <= tgt
            g = 0 if ok else (v - tgt) / tgt
        met += ok
        gap += g
    return met, gap, vals


def main():
    out = {}
    for name, members in CANDS.items():
        cfg = copy.deepcopy(BASE)
        cfg["members"] = members
        try:
            df, W, rep = EN.run(cfg, {"dev": ("2022-01-01", "2025-01-01")})
        except FileNotFoundError as e:
            print(f"{name}: 성분 파일 없음 — {e}")
            continue
        met, gap, vals = score(rep["dev"])
        out[name] = {"members": members, "dev": vals, "dev_met": met, "dev_gap": round(gap, 4)}
        print(f"{name:<14} 학습(2022–24) 달성 {met}/5 · 남은 거리 {gap:.3f} · {vals}")
    best = sorted(out, key=lambda k: (-out[k]["dev_met"], out[k]["dev_gap"]))[0]
    print(f"\n→ 학습 구간 기준 선택: {best}")
    for name in dict.fromkeys(["A 현재", best]):
        cfg = copy.deepcopy(BASE)
        cfg["members"] = out[name]["members"]
        df, W, rep = EN.run(cfg, {"test": TEST})
        met, gap, vals = score(rep["test"])
        out[name]["test"], out[name]["test_met"] = vals, met
        print(f"  시험(2025–26) {name:<14} 달성 {met}/5 · {vals}")
    out["_chosen"] = best
    (OUT / "price_compare_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
