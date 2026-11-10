# -*- coding: utf-8 -*-
"""
build_dashboard.py — 앙상블 결과를 HTML 한 장에 담은 대시보드 생성
==============================================================================
  · ensemble.py --final, warnings_model.py --final, export_results.py 이후 실행
  · 결과 데이터를 HTML 안에 넣으므로 더블클릭만으로 열림(서버·인터넷 불필요)
  · 데이터가 들어간 파일이라 outputs/ 에만 저장 — 저장소에 올리지 않음

실행
  python experiments/ensemble/build_dashboard.py
산출
  outputs/ensemble/dashboard.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT

HERE = Path(__file__).resolve().parent
START = "2024-01-01"
NAMES = {"chr_ft_lora": "Chronos-2", "lgb_stack_ft_weekly": "LightGBM 보정형(미세조정 Chronos 입력)",
         "lgb_stack_weekly": "LightGBM 보정형(제로샷 Chronos 입력)", "lgb_l1_weekly": "LightGBM 기본형"}


def arr(v):
    return [None if np.isnan(x) else round(float(x), 1) for x in v]


def main():
    p = pd.read_pickle(OUT / "price_final.pkl")
    last = p.loc[p["y"].notna(), "date"].max()                 # 실적이 있는 마지막 날까지
    p = p[(p["date"] >= START) & (p["date"] <= last)]
    days = []
    for d, g in p.groupby("date", sort=True):
        g = g.sort_values("hour")
        days.append({"d": d, "off": int(g["off"].iloc[0]), "y": arr(g["y"]), "p": arr(g["pred"]),
                     "lo": arr(g["lo"]), "hi": arr(g["hi"])})

    w = pd.read_csv(OUT / "price_weights.csv", encoding="utf-8-sig", index_col=0)
    order = [c for c in NAMES if c in w.columns]                     # 색 순서 고정 (Chronos-2 가 1번)
    w = w.loc[START[:7]:last[:7], order]
    weights = {"members": [NAMES[c] for c in w.columns],
               "rows": [{"ym": ym, "w": [round(float(x), 3) for x in r]} for ym, r in w.iterrows()]}

    wf = pd.read_pickle(OUT / "warnings_final.pkl")
    wf = wf[(wf["date"] >= START) & (wf["date"] <= last) & (wf["level"] != "") & wf["label"].isin(["crash", "spike25"])]
    warn = {}
    for r in wf.itertuples():
        key = ("c" if r.label == "crash" else "s") + r.stage
        warn.setdefault(r.date, {})[key] = [round(float(r.p), 3), r.level, None if np.isnan(r.y) else int(r.y)]
    hol = wf.drop_duplicates("date").set_index("date")["holiday_name"].dropna().to_dict()

    rep = json.load(open(OUT / "warnings_report.json", encoding="utf-8"))
    th = {("c" if "crash" in k else "s") + k[:2]: {"alarm": v["th_alarm"], "watch": v["th_watch"], "w": v["weights"]}
          for k, v in rep.items() if k.endswith("crash") or k.endswith("spike25")}

    targets = pd.read_csv(OUT / "targets_check.csv", encoding="utf-8-sig").fillna("").astype(str)
    data = {"days": days, "weights": weights, "warn": warn, "hol": hol, "th": th,
            "targets": {"cols": list(targets.columns), "rows": targets.values.tolist()}}
    html = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"저장 : {OUT / 'dashboard.html'}  ({len(html) / 1e6:.1f}MB, {len(days)}일)")


if __name__ == "__main__":
    main()
