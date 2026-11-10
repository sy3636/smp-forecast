# -*- coding: utf-8 -*-
"""
tune_dev2.py — 2차 개선 탐색 (개발 구간 2024~2025 전용, 2026년 자료는 읽기 전에 잘라 냄)
==============================================================================
  members : 성분 조합 × 가중 산출 창 × 가중 구분(전체 / 평일·휴일 / 평일·휴일×한낮)
  crash   : 급락 확률 반영 강도 α · 하한 floor   (warnings_model.py 로 warnings_final.pkl 을 먼저 만들 것)
  interval: 80% 범위 창·ACI 계수 재확인
실행
  python experiments/ensemble/tune_dev2.py members
  python experiments/ensemble/tune_dev2.py crash
  python experiments/ensemble/tune_dev2.py interval
"""
import itertools
import sys

import numpy as np

from common import DEV, OUT, fmt_metrics, interval_metrics, metrics4, period_mask
from ensemble import CONFIG, chronos_scale, combine, conformal, crash_adjust, load_members

CANDS = {
    "A 현재(연 1회 미세조정)": ["lgb_stack_weekly", "lgb_l1_weekly", "chr_ft_lora"],
    "B 분기 미세조정": ["lgb_stack_weekly", "lgb_l1_weekly", "chr_ft_lora_q"],
    "C A+적층(미세조정 입력)": ["lgb_stack_ft_weekly", "lgb_stack_weekly", "lgb_l1_weekly", "chr_ft_lora"],
    "D B+적층(미세조정 입력)": ["lgb_stack_ft_weekly", "lgb_stack_weekly", "lgb_l1_weekly", "chr_ft_lora_q"],
    "E 전부": ["lgb_stack_ft_weekly", "lgb_stack_weekly", "lgb_l1_weekly", "chr_ft_lora", "chr_ft_lora_q"],
}


def exists(m):
    return (OUT / (f"chronos_{m[4:]}.pkl" if m.startswith("chr_") else f"pred_{m}.pkl")).exists()


def dev(df, col):
    m = period_mask(df["date"], DEV)
    return metrics4(df.loc[m, "y"], df.loc[m, col], df.loc[m, "n1"])


def load(members):
    df = load_members(members)
    return df[df["date"] < DEV[1]].reset_index(drop=True)          # 2026년 자료 제거


def tune_members():
    allm = sorted({m for v in CANDS.values() for m in v if exists(m)})
    df = load(allm)
    print("성분별 (개발 2024~2025)")
    for m in allm:
        print(f"  {m:<22} {fmt_metrics(dev(df, m))}")
    rows = []
    for (name, mem), win, seg in itertools.product(CANDS.items(), [180, 365], [None, "off", "off_mid"]):
        if not all(exists(m) for m in mem):
            continue
        df["_e"], W = combine(df, mem, win, seg)
        r = dev(df, "_e")
        rows.append((r["MAE"], name, win, seg, r, W.loc["2024-01":"2025-12"].mean().round(2).to_dict()))
    rows.sort(key=lambda x: x[0])
    print("\n조합 × 창 × 구분 (MAE 순)")
    for mae, name, win, seg, r, w in rows[:12]:
        print(f"  {name:<22} 창 {win} 구분 {str(seg):<8} {fmt_metrics(r)}")
    print("\n최우수 평균 가중:", rows[0][5])


def tune_crash():
    df = load(CONFIG["members"])
    df["ens"], _ = combine(df, CONFIG["members"], CONFIG["weight_window"], CONFIG.get("segment"))
    base = dev(df, "ens")
    print(f"보정 없음        {fmt_metrics(base)}")
    rows = []
    for a, fl in itertools.product([0.25, 0.5, 0.75, 1.0, 1.25], [0.0, 30.0]):
        df["_c"] = crash_adjust(df, "ens", a, fl)
        r = dev(df, "_c")
        rows.append((r["MAE"], a, fl, r))
    rows.sort(key=lambda x: x[0])
    for mae, a, fl, r in rows:
        print(f"α {a:<4} 하한 {fl:>4.0f}  {fmt_metrics(r)}")


def tune_interval():
    df = load(CONFIG["members"])
    df["ens"], _ = combine(df, CONFIG["members"], CONFIG["weight_window"], CONFIG.get("segment"))
    ca = CONFIG.get("crash_adjust") or {}
    df["pred"] = crash_adjust(df, "ens", ca.get("alpha", 0.0), ca.get("floor", 0.0))
    m = period_mask(df["date"], DEV)
    sc = chronos_scale(df, "dem_off")
    mon = df.loc[m, "date"].str[:7]
    for W, eta in itertools.product([56, 91, 182], [0, 0.01, 0.02, 0.03]):
        lo, hi = conformal(df, "pred", W, sc, eta=eta)
        r = interval_metrics(df.loc[m, "y"], lo[m], hi[m])
        cov = [interval_metrics(df.loc[m, "y"][mon == ym], lo[m][mon == ym], hi[m][mon == ym])["cover"]
               for ym in sorted(mon.unique())]
        print(f"W {W:>3} η {eta:<4} 포함 {r['cover']:5.1f}% (월별 {min(cov):.0f}~{max(cov):.0f}%, "
              f"75~85% 밖 {sum(c < 75 or c > 85 for c in cov)}개월)  폭 {r['width']:5.1f}")


if __name__ == "__main__":
    {"members": tune_members, "crash": tune_crash, "interval": tune_interval}[sys.argv[1]]()
