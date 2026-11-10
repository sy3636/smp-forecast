# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step4 다단계 예측 지평 성능 측정
================================================
배경 : SMP는 하루전시장에서 D-1에 결정되며, 공공데이터포털 API가 D-1 저녁부터
       D일 24시간 확정값을 제공함(2026.8.11 실증 — 8/12분이 8/11 22:09 수집)
       → D+1 예측은 이미 공개된 값을 예측하는 구조로, 실용 가치가 제한적임
       본 스크립트는 **예측 가치가 발생하는 지평(D+2 이상)** 의 성능을 측정함

목적 : 예측 지평 H(1~7일)별 성능 곡선을 산출하여 실용 지평을 판정

방법
  ① 타깃 시프트 : y_H(t) = y(t + 24×(H-1))
     · 피처는 lag24 이상만 사용하므로, 타깃을 앞당기면 상대 시차가 24×H로 확대
     · 학습–운영 정합성(train-serving skew) 자동 확보
  ② 수요예측 변수 : H≥2 구간은 예측 시점에 미확보 → 자동 제외
  ③ 벤치마크 : 지속성 모형(직전 확보 가능 시점 값) 대비 개선율 산출

판정 기준
  · 실용 : 벤치마크 대비 개선율 10% 이상
  · 한계 : 개선율 0~10% — 참고 수준
  · 무의미 : 개선율 0% 이하 — 지속성 모형이 우수

입력 : data/processed/smp_features_{타깃}.csv
       data/processed/feature_dictionary_{타깃}.csv
출력 : logs/step4_horizon_{타깃}_{일시}.csv
       outputs/horizon/horizon_curve_{타깃}.png

버전 : v1.1 (2026-08-12) — 수요예측 사용/미사용 시나리오 비교
실행 : python scripts\\step4_horizon.py --target smp_land
       python scripts\\step4_horizon.py --target smp_land --max-h 7
"""

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

try:
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
except ImportError:
    plt = None

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "horizon")

TRAIN_START = "2021-01-01"
CV_END = "2025-12-31"
HOLD_START = "2026-01-01"
MAPE_FLOOR = 20.0                      # 저가 구간 발산 방지
EXCLUDE = {"date", "hour", "daytype", "year"}
C1, C2, C3 = "#1F3864", "#C00000", "#2E9E5B"

REPORT = []


def is_num(s):
    """수치형 컬럼 판정 — pandas 3.0 대응.

    pandas 3.0부터 문자열 컬럼 dtype이 object가 아닌 str(StringDtype)로 산출되어
    기존 'dtype != object' 필터가 문자열을 걸러내지 못함(2026.8.17 결함 조치).
    수치형 여부를 직접 판정하여 버전 무관하게 동작하도록 함.
    """
    return pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def rep(구분, 항목, 값, 비고=""):
    REPORT.append({"구분": 구분, "항목": 항목, "값": str(값), "비고": 비고})
    print(f"  · {항목} : {값}" + (f"  ({비고})" if 비고 else ""))


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    if not m.any():
        return None
    return float(np.mean(np.abs(p[m] - a[m]) / np.clip(a[m], MAPE_FLOOR, None)) * 100)


def rmse(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    return float(np.sqrt(np.mean((p[m] - a[m]) ** 2))) if m.any() else None


def make_model(seed=42):
    """LightGBM 우선, 미설치 시 scikit-learn 대체"""
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=600, learning_rate=0.05,
            num_leaves=63, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, verbose=-1), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=seed), "sklearn"


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="예측 지평별 성능 측정")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--max-h", type=int, default=7, help="최대 예측 지평(일)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="both", choices=["both", "with", "without"],
                    help="수요예측 변수 사용 시나리오 "
                         "(both=양자 비교 / with=사용 / without=미사용)")
    args = ap.parse_args()
    t = args.target

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"

    print("=" * 78)
    print(" step4 예측 지평별 성능 측정 (v1.0)")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 타깃 : {t}")
    print("=" * 78)

    src = os.path.join(PROCESSED, f"smp_features_{t}.csv")
    dic = os.path.join(PROCESSED, f"feature_dictionary_{t}.csv")
    for p in (src, dic):
        if not os.path.exists(p):
            sys.exit(f"[중단] 입력파일 없음 → {p}\n"
                     f"       python scripts\\step3_features.py --target {t} 선행 필요")

    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    df = df.sort_values(["date", "hour_n"]).reset_index(drop=True)
    dd = pd.read_csv(dic, encoding="utf-8-sig")

    print("\n■ ① 변수 구성")
    ok = set(dd[dd["운영가용"] == "Y"]["변수명"])
    base_feats = [c for c in df.columns if c in ok and c not in EXCLUDE]
    base_feats = [c for c in base_feats if is_num(df[c])]
    demand = [c for c in base_feats if c.startswith("demand_fcst")]
    rep("변수", "운영가용 전체", f"{len(base_feats)}개")
    rep("변수", "수요예측 계열", f"{len(demand)}개",
        "D+2 이상 구간에서는 미확보 → 제외")

    model, kind = make_model(args.seed)
    rep("설정", "학습 엔진", kind)
    rep("설정", "평가 구간",
        f"학습 {TRAIN_START}~{CV_END} / 검증 {HOLD_START}~")

    # ── 시나리오 정의 ──────────────────────────
    #  · with    : 수요예측 변수 사용 — D+1 확정 공개(23시) 이후 시점에만 가능
    #  · without : 수요예측 변수 미사용 — SMP·수요예측 공개 이전(15시 이전) 시점
    #               2026.8.12 15:49 관측에서 D+1 미제공 확인 → 해당 시점 예측 시나리오
    #  · D+2 이상은 어느 시각이든 수요예측 미확보이므로 without만 유효
    SCEN = {"both": ["with", "without"], "with": ["with"], "without": ["without"]}
    scens = SCEN[args.scenario]
    rep("설정", "측정 시나리오", " / ".join(scens),
        "with=수요예측 사용(23시 이후) / without=미사용(15시 이전)")

    # ── 지평별 학습·평가 ────────────────────────
    print(f"\n■ ② 지평별 성능 ({args.max_h}단계)")
    print(f"  {'지평':<7}{'시나리오':<10}{'변수':>5}{'학습행':>9}{'검증행':>8}"
          f"{'MAPE':>9}{'RMSE':>8}{'지속성':>9}{'개선율':>9}{'판정':>8}")
    print("  " + "-" * 84)

    rows = []
    for h in range(1, args.max_h + 1):
      for sc in scens:
        if h >= 2 and sc == "with":
            continue                      # D+2 이상은 수요예측 확보 불가
        shift = 24 * (h - 1)              # H=1은 현행 구조와 동일
        d = df.copy()
        d["_y"] = d[t].shift(-shift)      # 타깃을 H-1일 앞으로
        # 지속성 벤치마크 : 예측 시점에 확보 가능한 최근 동시간 실적
        #  · 타깃이 y(t+24(H-1))이므로 lag_d1(=y(t-24))이 곧 24H시간 전 값
        d["_naive"] = d["smp_lag_d1"] if "smp_lag_d1" in d.columns else d[t].shift(24)
        feats = (base_feats if sc == "with"
                 else [c for c in base_feats if c not in demand])

        tr = d[(d["date"] >= TRAIN_START) & (d["date"] <= CV_END)].dropna(subset=["_y"])
        ho = d[d["date"] >= HOLD_START].dropna(subset=["_y"])
        if len(tr) < 5000 or len(ho) < 500:
            continue

        m, _ = make_model(args.seed)
        m.fit(tr[feats], tr["_y"])
        pred = m.predict(ho[feats])

        mp, rm = mape(ho["_y"], pred), rmse(ho["_y"], pred)
        nv = mape(ho["_y"], ho["_naive"])
        if mp is None or nv is None:
            print(f"  D+{h:<6} 평가 불가 — 유효 표본 부족")
            continue
        imp = (nv - mp) / nv * 100
        verdict = ("실용" if imp is not None and imp >= 10
                   else "한계" if imp is not None and imp > 0 else "무의미")
        lab_sc = "수요예측 사용" if sc == "with" else "미사용"
        print(f"  D+{h:<5}{lab_sc:<10}{len(feats):>5}{len(tr):>9,}{len(ho):>8,}"
              f"{mp:>8.2f}%{rm:>8.2f}{nv:>8.2f}%{imp:>8.1f}%{verdict:>8}")
        rows.append({"지평": f"D+{h}", "H": h, "시나리오": lab_sc, "변수수": len(feats),
                     "학습행": len(tr), "검증행": len(ho),
                     "MAPE": round(mp, 2), "RMSE": round(rm, 2),
                     "지속성MAPE": round(nv, 2), "개선율": round(imp, 1),
                     "판정": verdict})
    print("  " + "-" * 84)

    if not rows:
        sys.exit("[중단] 평가 가능한 지평이 없음 — 표본 부족")

    r = pd.DataFrame(rows)
    best = r[r["판정"] == "실용"]["H"].max() if (r["판정"] == "실용").any() else 0
    rep("판정", "실용 지평 한계", f"D+{int(best)}" if best else "없음",
        "벤치마크 대비 개선율 10% 이상 구간")

    # 운영 시각 판단 — D+1 수요예측 사용/미사용 대비
    d1w = r[(r["H"] == 1) & (r["시나리오"] == "수요예측 사용")]
    d1o = r[(r["H"] == 1) & (r["시나리오"] == "미사용")]
    d2 = r[r["H"] == 2]
    if not d1o.empty:
        rep("판정", "D+1 미사용 MAPE", f"{d1o['MAPE'].iloc[0]:.2f}%",
            "SMP 공개 이전(15시 이전) 시점 예측 성능")
        if not d1w.empty:
            rep("판정", "수요예측 기여도",
                f"{d1o['MAPE'].iloc[0] - d1w['MAPE'].iloc[0]:+.2f}%p",
                "미사용 대비 사용 시 개선폭")
        if not d2.empty:
            rep("판정", "D+1(미사용) vs D+2",
                f"{d2['MAPE'].iloc[0] - d1o['MAPE'].iloc[0]:+.2f}%p",
                "시각 전진 시 확보 가능한 정확도 이득")

    # ── 그래프 ─────────────────────────────────
    if plt is not None and len(r) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(9, 4.6))
            ro = r[r["시나리오"] == "미사용"]
            rw = r[r["시나리오"] == "수요예측 사용"]
            ax.plot(ro["H"], ro["MAPE"], marker="o", lw=2.2, color=C1,
                    label="Model (no demand forecast)")
            if not rw.empty:
                ax.plot(rw["H"], rw["MAPE"], marker="D", ms=9, lw=0,
                        color=C3, label="Model (with demand forecast)")
            ax.plot(ro["H"], ro["지속성MAPE"], marker="s", lw=1.8, ls="--",
                    color=C2, label="Persistence benchmark")
            ax.axhline(8.0, color=C3, ls=":", lw=1.5, label="Gate 3 (8.0%)")
            ax.set_xlabel("Forecast horizon (days ahead)")
            ax.set_ylabel("MAPE (%)")
            ax.set_title(f"Forecast horizon vs accuracy — {t}",
                         fontsize=12, weight="bold")
            ax.set_xticks(sorted(r["H"].unique()))
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            pp = os.path.join(OUT_DIR, f"horizon_curve_{t}.png")
            fig.savefig(pp, dpi=120)
            plt.close(fig)
            print(f"\n  → 그래프 {pp}")
        except Exception as e:
            print(f"  [건너뜀] 그래프 생성 실패 — {str(e)[:60]}")

    # ── 저장 ───────────────────────────────────
    lp = os.path.join(LOG_DIR, f"step4_horizon_{t}_{ts}.csv")
    with open(lp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["구분", "항목", "값", "비고"])
        for x in REPORT:
            w.writerow([x["구분"], x["항목"], x["값"], x["비고"]])
        w.writerow([])
        w.writerow(list(r.columns))
        for _, x in r.iterrows():
            w.writerow(list(x.values))
    print(f"  → 검증보고 {lp}")

    print("=" * 78)
    if best >= 2:
        print(f" 결론 : D+{int(best)}까지 예측 실용성 확보 — 지평 확대 채택 검토")
    else:
        print(" 결론 : D+2 이상 실용성 미확보 — 지속성 모형 대비 개선 제한적")
    print("=" * 78)


if __name__ == "__main__":
    main()
