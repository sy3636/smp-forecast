# -*- coding: utf-8 -*-
"""
build_overview.py — 과제 1·2·3 + 급등을 한 장에 담은 「한눈에 보기」 대시보드
==============================================================================
  · ensemble.py --final, warnings_model.py --final, explain.py 이후 실행
  · 지정한 기간(기본 2026-01-01~10-06)의 날짜만 담고, 성적표도 그 기간으로 다시 계산
  · 결과 데이터를 HTML 안에 넣으므로 더블클릭만으로 열림(서버·인터넷 불필요)
  · 데이터가 들어간 파일이라 outputs/ 에만 저장 — 저장소에 올리지 않음

실행
  python experiments/ensemble/build_overview.py
  python experiments/ensemble/build_overview.py --start 2026-01-01 --end 2026-11-02
산출
  outputs/ensemble/overview.html
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dashboard import NAMES, arr
from common import OUT, interval_metrics, metrics4
from explain import GROUPS, OTHER
from export_results import REFERENCE, TARGETS, gap, judge

HERE = Path(__file__).resolve().parent
# 급락은 전날 15시 판단, 급등은 날짜마다 1회(전날 21시 판단 — 확정 수요예측 사용)
CRASH_STAGE, SPIKE_STAGE = "15", "21"
WARN_ROWS = [("crash15", "crash", CRASH_STAGE, "급락 (15시)"),
             ("spike25", "spike25", SPIKE_STAGE, "급등 +25%"), ("spike20", "spike20", SPIKE_STAGE, "급등 +20%"),
             ("spike30", "spike30", SPIKE_STAGE, "급등 +30%")]


def round_pct(v):
    """합이 100 이 되도록 큰 나머지 순으로 반올림"""
    raw = np.asarray(v, float) * 100
    fl = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - fl))[: 100 - fl.sum()]:
        fl[j] += 1
    return fl


def kdate(s):
    t = pd.Timestamp(s)
    return f"{t.year}. {t.month}. {t.day}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-10-06")
    args = ap.parse_args()
    a, b = args.start, args.end
    inside = lambda df: df[(df["date"] >= a) & (df["date"] <= b)]
    label = f"{kdate(a)}~{kdate(b)[6:] if a[:4] == b[:4] else kdate(b)}"

    p = pd.read_pickle(OUT / "price_final.pkl")
    p = inside(p[p["y"].notna()])
    days = [{"d": d, "off": int(g["off"].iloc[0]), "y": arr(g["y"]), "p": arr(g["pred"]), "lo": arr(g["lo"]), "hi": arr(g["hi"])}
            for d, g in p.sort_values("hour").groupby("date", sort=True)]

    # 가격 예측 목표 판정 (기간 전체)
    f, iv = metrics4(p["y"], p["pred"], p["n1"]), interval_metrics(p["y"], p["lo"], p["hi"])
    targets = []
    for name, goal, get, key, lo, hi, strict, unit, nd in TARGETS:
        v = float(get(f, iv))
        targets.append({"name": name, "goal": goal, "v": round(v, nd), "ok": judge(key(v), lo, hi, strict),
                        "gap": gap(key(v), lo, hi, strict, unit, nd)})
    ref = {k: round(float(iv[k]), nd) for _, k, nd in REFERENCE}
    rows = [[t["name"], t["goal"], t["v"], t["ok"], t["gap"]] for t in targets]
    rows += [[name, "참고", ref[k], "", ""] for name, k, _ in REFERENCE]
    ok = sum(t["ok"] == "달성" for t in targets)
    rows.append(["달성 개수", f"{len(targets)}개 중", "", f"{ok}개 달성", ""])

    # 가격 예측 비율 (기간에 걸친 달의 평균)
    w = pd.read_csv(OUT / "price_weights.csv", encoding="utf-8-sig", index_col=0)
    w = w.loc[a[:7]:b[:7], [c for c in NAMES if c in w.columns]].mean()
    pc = dict(zip(w.index, round_pct(w / w.sum())))
    lgb_parts = [c for c in w.index if c != "chr_ft_lora"]
    mix = (f"Chronos-2 {pc['chr_ft_lora']}% · LightGBM {sum(pc[c] for c in lgb_parts)}% "
           f"({' + '.join(NAMES[c].removeprefix('LightGBM ') + f' {pc[c]}' for c in lgb_parts)})")

    # 경보 — 날짜별 표시 + 기간 성적 (warnings_model.py 보고서와 같은 집계: 경보/주의 이상 × 실제)
    wf = inside(pd.read_pickle(OUT / "warnings_final.pkl"))
    wf = wf[wf["level"] != ""]
    warn = {}
    shown = (((wf["label"] == "crash") & (wf["stage"] == CRASH_STAGE))
             | ((wf["label"] == "spike25") & (wf["stage"] == SPIKE_STAGE)))
    for r in wf[shown].itertuples():
        key = "c" if r.label == "crash" else "s"
        warn.setdefault(r.date, {})[key] = [round(float(r.p), 3), r.level, None if np.isnan(r.y) else int(r.y)]
    hol = wf.drop_duplicates("date").set_index("date")["holiday_name"].dropna().to_dict()
    warn_table = []
    for key, lab, st, name in WARN_ROWS:
        x = wf[(wf["label"] == lab) & (wf["stage"] == st) & wf["y"].notna()]
        y, al, wa = x["y"] == 1, x["level"] == "경보", x["level"].isin(["주의", "경보"])
        warn_table.append({"key": key, "label": name, "days": len(x), "n": int(y.sum()), "tp": int((al & y).sum()),
                           "fp": int((al & ~y).sum()), "fn": int((~al & y).sum()), "wtp": int((wa & y).sum()), "wfp": int((wa & ~y).sum())})

    # 과제 3 근거
    E = inside(pd.read_pickle(OUT / "explain_daily.pkl"))
    exp = {r["date"]: {"s": r["sentence"], "c": [round(float(r[g]), 2) for g in GROUPS + [OTHER]]} for _, r in E.iterrows()}
    imp = E[GROUPS].abs().mean().sort_values(ascending=False)
    exp_rank = " · ".join(f"{g} {v:.1f}원" for g, v in imp.head(3).items())

    cards = {"targets": targets, "ok": ok, "width": ref["width"], "below": ref["below"], "above": ref["above"],
             "exp_days": len(exp), "exp_rank": exp_rank}
    data = {"period": {"start": a, "end": b, "label": label}, "days": days, "warn": warn, "hol": hol,
            "warnTable": warn_table, "exp": exp, "groups": GROUPS + [OTHER], "cards": cards, "mix": mix,
            "targets": {"cols": ["지표", "목표", label, "판정", "목표 대비"], "rows": rows}}
    html = (HERE / "overview_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "overview.html").write_text(html, encoding="utf-8")
    print(f"저장 : {OUT / 'overview.html'}  ({len(html) / 1e6:.1f}MB, {len(days)}일, {label})")
    print("비율 :", mix)
    print("목표 :", " · ".join(f"{t['name']} {t['v']} {t['ok']}" for t in targets))
    for r in warn_table:
        print(f"  {r['label']:<16} 실제 {r['n']:>2} · 경보 적중 {r['tp']:>2} · 헛경보 {r['fp']:>3} · 놓침 {r['fn']} · 주의 포함 {r['wtp']}/{r['wfp']}")


if __name__ == "__main__":
    main()
