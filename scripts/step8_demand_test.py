# -*- coding: utf-8 -*-
"""
step8_demand_test.py — 수요예측 투입에 의한 오차 저감 실증
==============================================================================
목적
  · 시간별 운영(23:30)은 수요예측을 이미 사용함
  · 주간 운영(14~15시)은 D+1 수요예측 미공개로 제외함
  · 「수요예측을 넣으면 오차가 줄어드는가」를 G-1 홀드아웃 2구간으로 재확인

시나리오
  · 미사용 : 운영가용 변수에서 demand_fcst* · net_load_fcst 제외
             (14시 주간 예측과 동일한 정보집합)
  · 사용   : 위 변수 포함 (23시 시간별 예측과 동일한 정보집합)
             ※ 14시에 이 값을 쓰는 것은 정보 누설 → 상한(이론)으로만 해석

판정 (G-1 준용)
  · 전체 MAPE 0.10%p 이상 개선, 홀드아웃 2구간 부호 일관
  · 주지표 MAE도 병기 (목표오차 확정서)

산출
  · data/processed/demand_test.csv
  · data/processed/demand_test_0901.csv
  · logs/step8_demand_test_{일시}.csv

실행
  python scripts\\step8_demand_test.py
"""
import os
import sys
import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("[중단] lightgbm 미설치")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SRC = os.path.join(PROCESSED, "smp_features_smp_land.csv")
DICT = os.path.join(PROCESSED, "feature_dictionary_smp_land.csv")
OUT = os.path.join(PROCESSED, "demand_test.csv")
OUT_0901 = os.path.join(PROCESSED, "demand_test_0901.csv")

TARGET = "smp_land"
MAPE_FLOOR = 20.0
GATE_MAPE = 0.10
EXCLUDE = {"date", "ts", "hour", "smp_land", "smp_jeju", "year",
           "unit_id", "unit_name", "grp", "pv_source", "holiday_name"}
HOLDOUTS = [("2026-01~08", "2026-01-01", "2026-09-01"),
            ("2025-09~12", "2025-09-01", "2026-01-01")]
SEED = 42
_rows_log = []


def rep(항목, 값, 비고=""):
    print(f"  · {항목:<28} {값:<28} {비고}", flush=True)
    _rows_log.append({"항목": 항목, "값": 값, "비고": 비고})


def mape(pred, act):
    act = np.asarray(act, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = ~(np.isnan(act) | np.isnan(pred))
    if not m.any():
        return None
    return float(np.mean(np.abs(pred[m] - act[m]) / np.clip(act[m], MAPE_FLOOR, None)) * 100)


def mae(pred, act):
    act = np.asarray(act, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = ~(np.isnan(act) | np.isnan(pred))
    if not m.any():
        return None
    return float(np.abs(pred[m] - act[m]).mean())


def is_demand_col(c):
    return c.startswith("demand_fcst") or c == "net_load_fcst"


def fit_predict(tr, te, feats):
    cats = [c for c in ("month", "dow", "hour_n", "daytype", "season") if c in feats]
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    for c in feats:
        if str(Xtr[c].dtype) in ("string", "str") or Xtr[c].dtype == object:
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    m = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
        random_state=SEED, verbose=-1)
    m.fit(Xtr, tr[TARGET])
    return m.predict(Xte), m


def scores(pred, act, hour, is_wknd):
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    day = (hour >= 9) & (hour <= 21)
    wd_day = day & (is_wknd == 0)
    out = {
        "전체MAPE": mape(pred, act),
        "전체MAE": mae(pred, act),
        "주간MAE_9_21": mae(pred[day], act[day]),
        "평일주간MAE": mae(pred[wd_day], act[wd_day]) if wd_day.any() else None,
        "잔차": float((act - pred).mean()),
    }
    return out


def main():
    print("=" * 72)
    print(" step8_demand_test — 수요예측 투입 실증 v1.0")
    print("=" * 72)

    print("\n■ ① 데이터")
    dic = pd.read_csv(DICT, encoding="utf-8-sig")
    key = "변수명" if "변수명" in dic.columns else dic.columns[0]
    ops = set(dic[dic["운영가용"] == "Y"][key])
    d = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    d = d.sort_values(["date", "hour_n"]).reset_index(drop=True)
    d = d[d[TARGET] > 1].copy()
    feats_all = [c for c in d.columns if c in ops and c not in EXCLUDE]
    demand = [c for c in feats_all if is_demand_col(c)]
    feats_no = [c for c in feats_all if c not in demand]
    # net_load_fcst가 N이어도 상한 시나리오에 포함
    extra = [c for c in ("net_load_fcst",) if c in d.columns and c not in demand]
    feats_yes = feats_all + extra
    feats_yes = list(dict.fromkeys(feats_yes))
    demand = [c for c in feats_yes if is_demand_col(c)]
    if "is_weekend" not in d.columns:
        d["is_weekend"] = pd.to_datetime(d["date"]).dt.dayofweek.isin([5, 6]).astype(int)
    dt = d["date"].astype(str)
    for c in ("month", "dow", "hour_n", "daytype", "season"):
        if c in d.columns:
            d[c] = d[c].astype("category")
    rep("행", f"{len(d):,}", f"{dt.min()} ~ {dt.max()}")
    rep("수요예측 계열", f"{len(demand)}종", ", ".join(demand[:8])
        + ("…" if len(demand) > 8 else ""))
    rep("미사용 변수", f"{len(feats_no)}종", "14시 정보집합")
    rep("사용 변수", f"{len(feats_yes)}종", "23시 정보집합")

    rows = []
    print("\n■ ② 홀드아웃 비교")
    hourly_0901 = None
    for lab, tr_end, te_end in HOLDOUTS:
        m_tr = (dt < tr_end).values
        m_te = (dt >= tr_end).values & (dt < te_end).values
        tr, te = d[m_tr], d[m_te]
        act = te[TARGET].values
        hour = te["hour_n"].astype(int).values
        wk = te["is_weekend"].astype(int).values
        print(f"\n  [{lab}] 학습 {len(tr):,} / 검증 {len(te):,}", flush=True)
        preds = {}
        for tag, feats in (("미사용", feats_no), ("사용", feats_yes)):
            p, model = fit_predict(tr, te, feats)
            preds[tag] = p
            sc = scores(p, act, hour, wk)
            rec = {"홀드아웃": lab, "조건": tag, "변수수": len(feats),
                   **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()}}
            rows.append(rec)
            print(f"    {tag:<6} MAPE {sc['전체MAPE']:.2f}%  MAE {sc['전체MAE']:.2f}원"
                  f"  9-21 MAE {sc['주간MAE_9_21']:.2f}원"
                  f"  평일주간 {sc['평일주간MAE']:.2f}원", flush=True)
            if tag == "사용" and hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=feats)
                dem_imp = imp[imp.index.isin(demand)]
                share = 100 * dem_imp.sum() / imp.sum() if imp.sum() else 0
                top = dem_imp.sort_values(ascending=False).head(5)
                rec["수요중요도합(%)"] = round(share, 2)
                rows[-1] = rec
                rep(f"[{lab}] 수요 중요도", f"{share:.1f}%",
                    " / ".join(f"{k} {100 * v / imp.sum():.1f}%" for k, v in top.items()))

        # 9/1이 2026 홀드아웃 밖(9/1 == te_end) — 별도 평가
        if lab.startswith("2026"):
            m1 = dt == "2026-09-01"
            if m1.any():
                te1 = d[m1]
                tr1 = d[dt < "2026-09-01"]
                act1 = te1[TARGET].values
                h1 = te1["hour_n"].astype(int).values
                w1 = te1["is_weekend"].astype(int).values
                recs_h = te1[["date", "hour_n", TARGET, "demand_fcst_land"]].copy()
                print("\n  [2026-09-01 단독] 학습 < 당일", flush=True)
                for tag, feats in (("미사용", feats_no), ("사용", feats_yes)):
                    p, _ = fit_predict(tr1, te1, feats)
                    recs_h[f"예측_{tag}"] = np.round(p, 2)
                    sc = scores(p, act1, h1, w1)
                    rec = {"홀드아웃": "2026-09-01", "조건": tag, "변수수": len(feats),
                           **{k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in sc.items()}}
                    rows.append(rec)
                    day = (h1 >= 9) & (h1 <= 21)
                    print(f"    {tag:<6} 전일 MAE {sc['전체MAE']:.2f}원"
                          f"  9-21 MAE {mae(p[day], act1[day]):.2f}원"
                          f"  9-21 잔차 {(act1[day] - p[day]).mean():+.2f}원",
                          flush=True)
                recs_h["잔차_미사용"] = recs_h[TARGET] - recs_h["예측_미사용"]
                recs_h["잔차_사용"] = recs_h[TARGET] - recs_h["예측_사용"]
                hourly_0901 = recs_h

    e = pd.DataFrame(rows)
    os.makedirs(PROCESSED, exist_ok=True)
    e.to_csv(OUT, index=False, encoding="utf-8-sig")
    if hourly_0901 is not None:
        hourly_0901.to_csv(OUT_0901, index=False, encoding="utf-8-sig")

    print("\n■ ③ 9/1 수요 vs 가격 (전일 대비)")
    a = d[dt.isin(["2026-08-31", "2026-09-01"])]
    if not a.empty and "demand_fcst_land" in a.columns:
        for h in (9, 12, 15, 19):
            r = a[a["hour_n"].astype(int) == h]
            if len(r) < 2:
                continue
            d0 = r[dt[r.index] == "2026-08-31"].iloc[0]
            d1 = r[dt[r.index] == "2026-09-01"].iloc[0]
            print(f"    {h:>2}시  수요 {d0['demand_fcst_land']:.0f}→{d1['demand_fcst_land']:.0f} MW"
                  f" ({d1['demand_fcst_land'] - d0['demand_fcst_land']:+.0f})"
                  f"  SMP {d0[TARGET]:.1f}→{d1[TARGET]:.1f}원"
                  f" ({d1[TARGET] - d0[TARGET]:+.1f})")

    print("\n■ ④ 판정")
    base = e[e["홀드아웃"].isin([h[0] for h in HOLDOUTS])]
    deltas = []
    for lab, _, _ in HOLDOUTS:
        a = base[(base["홀드아웃"] == lab) & (base["조건"] == "미사용")].iloc[0]
        b = base[(base["홀드아웃"] == lab) & (base["조건"] == "사용")].iloc[0]
        dm = a["전체MAPE"] - b["전체MAPE"]
        da = a["전체MAE"] - b["전체MAE"]
        deltas.append(dm)
        mark = "개선" if dm >= GATE_MAPE else ("악화" if dm < 0 else "미달")
        print(f"  [{lab}] MAPE {a['전체MAPE']:.2f}→{b['전체MAPE']:.2f}% "
              f"({dm:+.2f}%p)  MAE {a['전체MAE']:.2f}→{b['전체MAE']:.2f}원 "
              f"({da:+.2f}원)  {mark}")
        rep(f"[{lab}] MAPE 증감", f"{dm:+.2f}%p", mark)
    if all(x >= GATE_MAPE for x in deltas):
        verdict = "채택 — 23시 시간별 경로에서 수요예측 유지"
    elif all(x > 0 for x in deltas) and min(deltas) < GATE_MAPE:
        verdict = "유지(약한 개선) — 시간별 유지, 주간 14시 투입은 누설"
    elif all(x < 0 for x in deltas):
        verdict = "기각 — 수요예측 투입이 양 구간 악화"
    else:
        verdict = "기각(부호 역전) — 주간 14시 신규 투입 불가, 시간별은 현행 유지 여부 별도"
    # 시간별은 이미 사용 중. 주간 신규 투입은 14시 미확보 + 누설.
    print(f"\n  G-1 : {verdict}")
    print("  운영 : 주간(14시) 수요예측 미사용 유지 — D+1 수요는 해당 시각 미공표")
    print("         시간별(23시) 수요예측 사용은 정보집합이 성립할 때의 효과로 해석")
    rep("판정", verdict, "G-1 0.10%p · 양 홀드아웃")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lp = os.path.join(LOG_DIR, f"step8_demand_test_{ts}.csv")
    pd.DataFrame(_rows_log).to_csv(lp, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {os.path.relpath(OUT, BASE_DIR)}")
    print(f"       {os.path.relpath(lp, BASE_DIR)}")


if __name__ == "__main__":
    main()
