# -*- coding: utf-8 -*-
"""
export_results.py — 발표·대시보드용 CSV 내보내기 (ensemble.py --final, warnings_model.py --final 이후 실행)
==============================================================================
산출 (outputs/ensemble/)
  · price_predictions.csv — 시간별 실적·예측·80% 하한·상한·전일 동시간·성분별 예측 (2023-01~)
  · warnings_daily.csv    — 일별 급락·급등 확률·단계(주의/경보)·실제·월초(1~3일) 표시
  · targets_check.csv     — 지표별 목표·값·판정(달성/미달)·시험 목표 대비 차이 (개발·시험·부서 홀드아웃)
"""
import json

import pandas as pd

from common import OUT
from ensemble import CONFIG

WD = "월화수목금토일"


def main():
    p = pd.read_pickle(OUT / "price_final.pkl")
    p = p[p["date"] >= "2023-01-01"]
    cols = {"date": "date", "hour": "hour", "y": "실적", "pred": "예측", "lo": "하한80", "hi": "상한80", "n1": "전일동시간"}
    out = p[list(cols) + CONFIG["members"]].rename(columns=cols)
    out.round(2).to_csv(OUT / "price_predictions.csv", index=False, encoding="utf-8-sig")

    w = pd.read_pickle(OUT / "warnings_final.pkl")
    # 급락은 전날 15시 판단만, 급등은 날짜마다 1회(전날 21시 판단)
    w = w[(w["level"] != "") & (w["stage"] == w["label"].map(lambda l: "15" if l == "crash" else "21"))].copy()
    w["요일"] = w["dow"].map(lambda d: WD[int(d)])
    w["판단시각"] = (w["stage"] + "시").where(w["label"] == "crash", "날짜 단위")
    w["과제"] = w["label"].map({"crash": "급락", "spike25": "급등+25%", "spike20": "급등+20%", "spike30": "급등+30%"})
    keep = {"판단시각": "판단시각", "과제": "과제", "date": "date", "요일": "요일", "holiday_name": "휴일명",
            "p": "확률", "level": "단계", "y": "실제", "mid_min": "한낮최저", "day_max": "일최고", "max_hour": "최고시각",
            "ref28": "직전28일기준", "first3": "월초1_3일", "p_logit": "p_로지스틱", "p_lgbm": "p_LightGBM",
            "p_price": "p_가격신호"}
    w[list(keep)].rename(columns=keep).round(4).to_csv(OUT / "warnings_daily.csv", index=False, encoding="utf-8-sig")

    r = json.load(open(OUT / "price_report.json", encoding="utf-8"))["report"]
    t = targets_table(r)
    t.to_csv(OUT / "targets_check.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))


# 목표 — 실행계획서 v1.7 도전 목표, 과제 ② 성공 기준
#   (지표, 목표 표기, 값 꺼내기, 판정에 쓸 값, 하한, 상한, 상한 미포함 여부, 단위, 소수 자리)
TARGETS = [
    ("MAE", "≤ 7.0원", lambda f, iv: f["MAE"], abs, None, 7.0, False, "원", 2),
    ("rMAE", "≤ 0.65", lambda f, iv: f["rMAE"], abs, None, 0.65, False, "", 4),
    ("MAPE", "< 5.0%", lambda f, iv: f["MAPE"], abs, None, 5.0, True, "%p", 2),
    ("편의", "절대값 ≤ 1.5%", lambda f, iv: f["bias"], abs, None, 1.5, False, "%p", 2),
    ("80% 범위 포함률", "75~85%", lambda f, iv: iv["cover"], abs, 75.0, 85.0, False, "%p", 1),
]
REFERENCE = [("80% 범위 평균 폭(원)", "width", 1), ("하한 아래 이탈(%)", "below", 1), ("상한 위 이탈(%)", "above", 1)]
PERIODS = [("dev", "개발 2020~2024", "개발"), ("test", "시험 2025-01~2026-11-02", "시험"),
           ("hold", "부서 홀드아웃 2026-01-29~09-02", "홀드아웃"), ("fresh", "새 시험 2026-10-03~11-02", "새 시험")]


def judge(v, lo, hi, strict):
    ok_hi = v < hi if strict else v <= hi
    return "달성" if ok_hi and (lo is None or v >= lo) else "미달"


def gap(v, lo, hi, strict, unit, nd):
    """목표 경계까지의 차이 — 초과분(미달) 또는 남은 여유(달성)"""
    if lo is not None and v < lo:
        return f"하한보다 {lo - v:.{nd}f}{unit} 낮음"
    if (v >= hi) if strict else (v > hi):
        return f"{v - hi:+.{nd}f}{unit} 초과"
    return f"{hi - v:.{nd}f}{unit} 여유" if lo is None else "범위 안"


def targets_table(r):
    rows = []
    for name, goal, get, key, lo, hi, strict, unit, nd in TARGETS:
        row = {"지표": name, "목표": goal}
        for per, label, short in PERIODS:
            if per not in r:
                continue
            v = get(r[per]["final"], r[per]["interval"])
            row[label] = round(v, nd)
            row[f"{short} 판정"] = judge(key(v), lo, hi, strict)
            if per == "test":
                row["시험 목표 대비"] = gap(key(v), lo, hi, strict, unit, nd)
        rows.append(row)
    for name, k, nd in REFERENCE:
        row = {"지표": name, "목표": "참고"}
        for per, label, _ in PERIODS:
            if per in r:
                row[label] = round(r[per]["interval"][k], nd)
        rows.append(row)
    t = pd.DataFrame(rows)
    judged = t["목표"] != "참고"
    summary = {"지표": "달성 개수", "목표": f"{judged.sum()}개 중"}
    for c in [c for c in t.columns if c.endswith("판정")]:
        summary[c] = f"{(t.loc[judged, c] == '달성').sum()}개 달성"
    t = pd.concat([t, pd.DataFrame([summary])], ignore_index=True)
    order = ["지표", "목표"]
    for per, label, short in PERIODS:
        if per in r:
            order += [label, f"{short} 판정"] + (["시험 목표 대비"] if per == "test" else [])
    return t[order].fillna("")


if __name__ == "__main__":
    main()
