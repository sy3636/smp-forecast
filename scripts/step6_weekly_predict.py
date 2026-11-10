# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step6 주간 예측 운영 (D+1 ~ D+6)
================================================
배경 : SMP 확정값은 D-1 15~23시 사이에 공개됨(2026.8.12 실측 — 15:49 미제공, 23:04 제공)
       → **공개 이전 시각(14~15시)에 실행**하면 D+1도 실질 예측 가치를 가짐
       그 시점에는 수요예측도 미확보이므로 수요예측 미사용 모델을 적용함

목적 : 지평별 모델(h1~h6)로 향후 6일 × 24시간 SMP를 일괄 예측

구조
  ① 피처 기준행 : **D+1 일자 행**을 사용
     · 학습 구조 y_H(t) = y(t + 24×(H-1))에서 t = D+1이므로 대상은 D+H
     · 시차 변수는 lag24 이상 → D+0 확정 실적으로 충족(예측 시각에 확보)
  ② 지평별 모델 적용 : ml/models/smp_{타깃}_h{H}.pkl
  ③ 제주 : HVDC 해저연계선으로 육지와 동일 SMP(2026.7 이후 일치율 100%)
     → 육지 예측값을 그대로 적용(사내 확인, 2026.8.11)

권고 운영 시각
  · 14~15시 : 본 스크립트 실행 → D+1~D+6 예측 제공
  · 23:30   : step6_predict.py 실행 → D+1 확정값 수집·사후검증

입력 : data/processed/smp_features_{타깃}.csv
       ml/models/smp_{타깃}_h{H}.pkl  (step4_train_horizon.py 산출)
출력 : outputs/predict/SMP주간예측_{기준일}.csv / .xlsx / .png
       logs/step6_weekly_history.csv

버전 : v1.0 (2026-08-12)
실행 : python scripts\\step6_weekly_predict.py
       python scripts\\step6_weekly_predict.py --no-rebuild   (데이터 갱신 생략)
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

try:
    import joblib
except ImportError:
    sys.exit("[중단] joblib 미설치 → pip install -r requirements.txt")

try:
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
except ImportError:
    plt = None

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HIST = os.path.join(LOG_DIR, "step6_weekly_history.csv")

CATEGORICAL = ["month", "dow", "hour_n"]
MAPE_FLOOR = 20.0
C1, C2 = "#1F3864", "#C00000"

STEPS = []


def log(단계, 상태, 내용=""):
    STEPS.append({"단계": 단계, "상태": 상태, "내용": 내용})
    mark = {"정상": "○", "주의": "△", "실패": "×", "생략": "-"}.get(상태, " ")
    print(f"  [{mark}] {단계} : {상태}" + (f" — {내용}" if 내용 else ""))


def _dec(b):
    """하위 프로세스 출력 디코딩 — utf-8 우선, 실패 시 cp949"""
    if not b:
        return ""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def run(script, args, timeout=1800):
    """하위 스크립트 실행 — 실패 시 stderr 요약 반환"""
    cmd = [sys.executable, os.path.join(SCRIPTS, script)] + args
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        out = [x for x in _dec(r.stdout).strip().split("\n") if x.strip()]
        err = [x for x in _dec(r.stderr).strip().split("\n") if x.strip()]
        if r.returncode == 0:
            return True, (out[-1] if out else "")
        msg = " | ".join(err[-3:]) if err else (out[-1] if out else "")
        return False, f"[종료코드 {r.returncode}] {msg[:300]}"
    except subprocess.TimeoutExpired:
        return False, f"타임아웃({timeout}초)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = ~(np.isnan(a) | np.isnan(p))
    if not m.any():
        return None
    return float(np.mean(np.abs(p[m] - a[m]) / np.clip(a[m], MAPE_FLOOR, None)) * 100)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="주간 SMP 예측 (D+1~D+6)")
    ap.add_argument("--target", default="smp_land", choices=["smp_land", "smp_jeju"])
    ap.add_argument("--max-h", type=int, default=6, help="최대 예측 지평(일)")
    ap.add_argument("--base", default=None,
                    help="기준일 YYYYMMDD(기본: 오늘) — D+1부터 예측")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="데이터 갱신·피처 생성 생략(최신 상태일 때)")
    args = ap.parse_args()
    t = args.target

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = datetime.now()
    base = args.base or f"{t0:%Y%m%d}"
    bd = datetime.strptime(base, "%Y%m%d")
    d1 = bd + timedelta(days=1)                      # 피처 기준행 = D+1

    print("=" * 78)
    print(" step6 주간 SMP 예측 — D+1 ~ D+{} (v1.0)".format(args.max_h))
    print(f" 실행시각 : {t0:%Y-%m-%d %H:%M:%S}")
    print(f" 기준일 : {bd:%Y-%m-%d} / 예측구간 : {d1:%Y-%m-%d} ~ "
          f"{bd + timedelta(days=args.max_h):%Y-%m-%d}")
    print("=" * 78)

    # ── ① 데이터 갱신 ──────────────────────────
    print("\n■ ① 데이터 갱신")
    if args.no_rebuild:
        log("데이터 갱신", "생략", "--no-rebuild 지정")
    else:
        end = f"{bd + timedelta(days=args.max_h):%Y%m%d}"
        for s, a, nm in (
            ("step1_collect.py", ["--source", "smp", "--start",
                                  f"{bd - timedelta(days=10):%Y%m%d}", "--end", end], "API 수집"),
            ("step2_preprocess.py", ["--source", "smp"], "전처리"),
            ("step1_collect_vilage_fcst.py", [], "단기예보 수집"),
            ("step8_pv_forecast.py", ["--apply", "--no-save"], "태양광 전망"),
            ("step3_build_dataset.py", ["--end", end], "데이터셋 통합"),
            ("step3_features.py", ["--target", t], "피처 생성"),
        ):
            okr, msg = run(s, a, 900)
            soft = {"API 수집", "단기예보 수집", "태양광 전망"}
            log(nm, "정상" if okr else ("주의" if nm in soft else "실패"), msg[:70])
            if not okr and nm not in soft:
                sys.exit("[중단] 데이터 갱신 실패")

    # ── ② 기준행 확보 ──────────────────────────
    print("\n■ ② 예측 기준행 확보")
    src = os.path.join(PROCESSED, f"smp_features_{t}.csv")
    if not os.path.exists(src):
        sys.exit(f"[중단] 피처셋 없음 → {src}")
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")

    key = f"{d1:%Y-%m-%d}"
    seg = df[df["date"] == key]
    if seg.empty:
        sys.exit(f"[중단] 기준행({key}) 없음 — 데이터셋 종료일 확인 필요")
    log("기준행", "정상", f"{key} / {len(seg)}시간")
    log("데이터 최종일", "정상", str(df["date"].max()))

    # ── ③ 지평별 예측 ──────────────────────────
    print("\n■ ③ 지평별 예측")
    print(f"  {'지평':<7}{'대상일':<13}{'평균':>9}{'최고(시)':>13}{'최저(시)':>13}"
          f"{'결측':>6}")
    print("  " + "-" * 62)

    rows, meta = [], []
    for h in range(1, args.max_h + 1):
        mp = os.path.join(MODEL_DIR, f"smp_{t}_h{h}.pkl")
        if not os.path.exists(mp):
            log(f"D+{h}", "실패", f"모델 없음 → {os.path.basename(mp)}")
            continue
        pack = joblib.load(mp)
        model, feats = pack["model"], pack["features"]
        miss = [c for c in feats if c not in seg.columns]
        if miss:
            log(f"D+{h}", "실패", f"변수 누락 {len(miss)}종: {miss[:3]}")
            continue

        na = int(seg[feats].isna().sum().sum())
        p = np.round(model.predict(seg[feats]), 2)
        tgt = bd + timedelta(days=h)
        try:
            from step6_wx_pattern import apply_pattern
            tseg = df[df["date"] == f"{tgt:%Y-%m-%d}"]
            src = tseg if len(tseg) == len(seg) else seg
            rain = src["rain_daytime"].values if "rain_daytime" in src.columns else 0
            sol = src["solar_ratio"].values if "solar_ratio" in src.columns else 1
            wk = src["is_weekend"].values if "is_weekend" in src.columns else 0
            lag = src["smp_lag_d1"].values if "smp_lag_d1" in src.columns else None
            pen = (src["pv_penetration"].values
                   if "pv_penetration" in src.columns else None)
            pv = src["pv_fcst"].values if "pv_fcst" in src.columns else None
            dem = (src["demand_fcst_land"].values
                   if "demand_fcst_land" in src.columns else None)
            p2 = apply_pattern(p, src["hour_n"].values, rain, sol, wk, lag,
                               pv_pen=pen, pv_fcst=pv, demand_fcst=dem)
            if not np.allclose(p, p2):
                log(f"D+{h} 패턴", "정상",
                    f"{tgt:%m/%d} 9.1형상 {p.mean():.1f}→{p2.mean():.1f}원")
            p = p2
            from step6_pv_pattern import apply_pv_softening
            p3, pnote = apply_pv_softening(
                p, src["hour_n"].values, pv_pen=pen, pv_fcst=pv,
                demand_fcst=dem, lag_d1=lag,
                light_load=(src["is_light_load_regime"].values
                            if "is_light_load_regime" in src.columns else None))
            if pnote and "없음" not in pnote and "생략" not in pnote:
                log(f"D+{h} 태양광", "정상", f"{tgt:%m/%d} {pnote}")
                p = p3
        except Exception as e:
            log(f"D+{h} 패턴", "주의", f"{type(e).__name__}")
        v = pd.Series(p)
        pk = int(seg["hour_n"].iloc[int(v.idxmax())])
        bt = int(seg["hour_n"].iloc[int(v.idxmin())])
        print(f"  D+{h:<5}{tgt:%Y-%m-%d}  {v.mean():>8.1f}"
              f"{v.max():>9.1f}({pk:>2}시){v.min():>9.1f}({bt:>2}시){na:>6}")

        for hh, pv in zip(seg["hour_n"].values, p):
            rows.append({"기준일": f"{bd:%Y-%m-%d}", "지평": f"D+{h}",
                         "대상일": f"{tgt:%Y-%m-%d}", "거래시간": int(hh),
                         "예측": float(pv)})
        meta.append({"지평": f"D+{h}", "H": h, "대상일": f"{tgt:%Y-%m-%d}",
                     "평균": round(float(v.mean()), 2),
                     "최고": round(float(v.max()), 2), "최고시": pk,
                     "최저": round(float(v.min()), 2), "최저시": bt,
                     "입력결측": na,
                     "홀드아웃MAPE": pack.get("holdout_mape"),
                     "학습시각": pack.get("trained_at")})
    print("  " + "-" * 62)

    if not rows:
        sys.exit("[중단] 예측 산출 실패 — "
                 f"python scripts\\step4_train_horizon.py --target {t} 선행 필요")

    out = pd.DataFrame(rows)
    mt = pd.DataFrame(meta)

    # ── ④ 실적 결합·사후검증 ───────────────────
    print("\n■ ④ 실적 결합")
    act = df[["date", "hour_n", t]].rename(
        columns={"date": "대상일", "hour_n": "거래시간", t: "실적"})
    out = out.merge(act, on=["대상일", "거래시간"], how="left")
    n_act = int(out["실적"].notna().sum())
    if n_act:
        v = out.dropna(subset=["실적"])
        m = mape(v["실적"], v["예측"])
        log("실적 확보", "정상", f"{n_act}시간 / 사후검증 MAPE {m:.2f}%")
    else:
        log("실적 확보", "생략", "예측 대상 구간 실적 미발생")

    # ── ⑤ 저장 ─────────────────────────────────
    print("\n■ ⑤ 산출물 저장")
    tag = f"{bd:%Y%m%d}"
    cp = os.path.join(OUT_DIR, f"SMP주간예측_{tag}.csv")
    tmp = cp + ".tmp"
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, cp)
    log("주간 예측표", "정상", cp)

    try:
        xp = os.path.join(OUT_DIR, f"SMP주간예측_{tag}.xlsx")
        with pd.ExcelWriter(xp, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="주간예측", index=False)
            mt.to_excel(w, sheet_name="지평별요약", index=False)
            piv = out.pivot_table(index="거래시간", columns="대상일",
                                  values="예측", aggfunc="first")
            piv.to_excel(w, sheet_name="시간×일자")
            pd.DataFrame({"내용": [
                "【주간 예측 활용 유의사항】", "",
                "1. 본 예측은 수요예측 변수 미사용 모델 산출값임",
                "   · 예측 시각(14~15시)에 D+1 수요예측이 미공개이기 때문",
                "",
                "2. D+1 확정값은 당일 14~20시 사이에 공개됨(2026.9월 실측, 공표 시각 변동)",
                "   · 공개 후에는 확정값을 우선 적용할 것",
                "   · 시간별 예측은 18:00·21:00·23:30 3단으로 산출하여 확정값을 반영함",
                "",
                "3. 지평이 멀수록 오차가 확대됨 — 홀드아웃 2026-01~09 실측",
                "   · MAE  : D+1 8.84원 → D+4 10.72원 → D+6 10.64원",
                "   · MAPE : D+1 8.70% → D+4 10.57% → D+6 10.48%",
                "",
                "4. 휴일 SMP가 낮더라도 휴일 작업은 근로조건·인력 확보 제약이 있으므로,",
                "   정비 일정은 안전·인사 부서 협의를 거쳐 결정할 것.",
                "   본 예측은 경제성 판단의 참고자료임.",
            ]}).to_excel(w, sheet_name="유의사항", index=False)
        log("Excel 산출", "정상", xp)
    except Exception as e:
        log("Excel 산출", "주의", f"{type(e).__name__}")

    if plt is not None:
        try:
            fig, ax = plt.subplots(figsize=(12, 4.8))
            for _, r in mt.iterrows():
                s = out[out["지평"] == r["지평"]]
                ax.plot(range(len(s)), s["예측"], lw=1.8,
                        label=f"{r['지평']} ({r['대상일'][5:]})")
            ax.set_xticks([i * 24 for i in range(len(mt) + 1)])
            ax.set_xticklabels([f"D+{i + 1}" for i in range(len(mt))] + [""])
            ax.set_xlabel("Forecast day (24h each)")
            ax.set_ylabel("SMP (won/kWh)")
            ax.set_title(f"Weekly SMP forecast from {bd:%Y-%m-%d}",
                         fontsize=12, weight="bold")
            ax.grid(alpha=0.3)
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            pp = os.path.join(OUT_DIR, f"SMP주간예측_{tag}.png")
            fig.savefig(pp, dpi=120)
            plt.close(fig)
            log("그래프", "정상", pp)
        except Exception as e:
            log("그래프", "주의", f"{type(e).__name__}")

    # 실행 이력 누적
    rec = {"실행시각": f"{t0:%Y-%m-%d %H:%M:%S}", "기준일": f"{bd:%Y-%m-%d}",
           "계통": t, "지평수": len(mt), "행수": len(out),
           "소요초": round((datetime.now() - t0).total_seconds(), 1)}
    for _, r in mt.iterrows():
        rec[f"평균_{r['지평']}"] = r["평균"]
    hd = not os.path.exists(HIST)
    pd.DataFrame([rec]).to_csv(HIST, mode="a", header=hd,
                               index=False, encoding="utf-8-sig")
    log("실행 이력", "정상", HIST)

    #  주간 예측 이력 누적·실적 재결합 (2026.9.5 신설)
    #   · 산출 시점에 미확보인 실적을 사후 재결합하여 지평별 잔차 이력을 확보
    #   · 주간 편의보정 확대(설계서 v1.0)의 선결 조건
    try:
        from step6_weekly_hist import rebuild as _hist_rebuild
        _h = _hist_rebuild(t, verbose=False)
        if _h is not None:
            _n = int(_h["실적"].notna().sum())
            log("예측 이력 누적", "정상",
                f"{len(_h):,}행 / 실적 결합 {_n:,}행")
        else:
            log("예측 이력 누적", "주의", "산출물 없음")
    except Exception as e:
        log("예측 이력 누적", "주의", f"{type(e).__name__}")

    print("=" * 78)
    print(f" 예측 완료 : {bd:%Y-%m-%d} 기준 {len(mt)}일 / 소요 "
          f"{(datetime.now() - t0).total_seconds():.1f}초")
    print(f"   구간 평균 {out['예측'].mean():.1f}원/kWh "
          f"(최고 {out['예측'].max():.1f} / 최저 {out['예측'].min():.1f})")
    print(" ※ D+1 확정값은 당일 14~20시 공개(실측, 변동) — 공개 후 확정값 우선 적용")
    print("=" * 78)


if __name__ == "__main__":
    main()
