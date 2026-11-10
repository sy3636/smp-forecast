# -*- coding: utf-8 -*-
"""
tune_dev.py — 개발 구간(2024~2025) 전용 설정 탐색 — 시험 구간(2026)은 읽지 않음
==============================================================================
  ① 성분별 성적  ② 가중 산출 창  ③ 성분 조합(하나씩 빼기)  ④ 편의보정 K·γ  ⑤ 80% 범위 방법·창
실행
  python experiments/ensemble/tune_dev.py [--members ...]
"""
import argparse
import itertools

import numpy as np

from common import DEV, interval_metrics, metrics4, period_mask, fmt_metrics
from ensemble import CONFIG, bias_correct, chronos_interval, chronos_scale, combine, conformal, load_members


def dev_metrics(df, col):
    m = period_mask(df["date"], DEV)
    return metrics4(df.loc[m, "y"], df.loc[m, col], df.loc[m, "n1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=CONFIG["members"])
    ap.add_argument("--extra", nargs="*", default=[], help="성분 후보로만 성적을 볼 추가 예측")
    args = ap.parse_args()
    members = args.members
    df = load_members(members + args.extra)
    df = df[df["date"] < DEV[1]].reset_index(drop=True)          # 시험 구간 제거
    print("① 성분별 (개발 2024~2025)")
    for c in members + args.extra:
        print(f"  {c:<16} {fmt_metrics(dev_metrics(df, c))}")

    print("\n② 가중 산출 창")
    best = None
    for win in [None, 365, 180, 90]:
        df["ens"], W = combine(df, members, win)
        r = dev_metrics(df, "ens")
        print(f"  창 {str(win):>5}  {fmt_metrics(r)}   가중(마지막 달) {np.round(W.iloc[-1].to_numpy(), 2)}")
        if best is None or r["MAE"] < best[0]:
            best = (r["MAE"], win)
    win = best[1]

    print(f"\n③ 하나씩 빼기 (창 {win})")
    df["ens"], W = combine(df, members, win)
    base = dev_metrics(df, "ens")["MAE"]
    for c in members:
        rest = [x for x in members if x != c]
        p, _ = combine(df, rest, win)
        df["_t"] = p
        print(f"  − {c:<14} MAE {dev_metrics(df, '_t')['MAE']:.3f} ({dev_metrics(df, '_t')['MAE'] - base:+.3f})")

    print("\n④ 편의보정 (K × γ, 일유형 분리 여부)")
    rows = []
    for K, g, split in itertools.product([7, 14, 21, 28], [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], [True, False]):
        df["_b"] = bias_correct(df, "ens", K, g, 25.0, split)
        r = dev_metrics(df, "_b")
        rows.append((r["MAE"], K, g, split, r))
    rows.sort(key=lambda x: x[0])
    for mae, K, g, split, r in rows[:8]:
        print(f"  K {K:>2} γ {g:<4} 분리 {split!s:<5} {fmt_metrics(r)}")
    print(f"  (무보정) {fmt_metrics(dev_metrics(df, 'ens'))}")
    _, K, g, split, _ = rows[0]
    df["pred"] = bias_correct(df, "ens", K, g, 25.0, split)

    print("\n⑤ 80% 범위")
    m = period_mask(df["date"], DEV)
    lo, hi = chronos_interval(df, "pred", "dem_off")
    r = interval_metrics(df.loc[m, "y"], lo[m], hi[m])
    print(f"  Chronos 분위수 그대로   포함 {r['cover']:5.1f}%  폭 {r['width']:5.1f}  아래 {r['below']:.1f}% 위 {r['above']:.1f}%")
    sc = chronos_scale(df, "dem_off")
    for meth, W, eta in [("plain", 56, 0), ("plain", 182, 0)] + list(
            itertools.product(["norm"], [56, 91, 182], [0, 0.005, 0.01, 0.02])):
        lo, hi = conformal(df, "pred", W, sc if meth == "norm" else None, eta=eta)
        meth = f"{meth} η{eta}"
        r = interval_metrics(df.loc[m, "y"], lo[m], hi[m])
        mon = df.loc[m, "date"].str[:7]
        cov = [interval_metrics(df.loc[m, "y"][mon == ym], lo[m][mon == ym], hi[m][mon == ym])["cover"]
               for ym in sorted(mon.unique())]
        print(f"  {meth:<11} W {W:>3}  포함 {r['cover']:5.1f}% (월별 {min(cov):.0f}~{max(cov):.0f}%, "
              f"75~85% 밖 {sum(c < 75 or c > 85 for c in cov)}개월)  폭 {r['width']:5.1f}  "
              f"아래 {r['below']:.1f}% 위 {r['above']:.1f}%")


if __name__ == "__main__":
    main()
