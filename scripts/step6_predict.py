# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - step6 운영 예측(익일 24시간)
================================================
목적 : 실운영 조건에서 익일 24시간대별 SMP를 예측하고 결과를 산출·보관
시점 : 매일 18:00(1차) · 21:00(2차) · 23:30(폴백) 3단 실행
       · D+1 공표 시각은 고정이 아님 — 2026.9.2~9.5 실측 14:00~20:00(6시간 편차)
       · 18시는 조기 공표일 대응(4일 중 2일 성공) — 실패해도 21시가 보완
       · 21시는 최장 관측치 20:00 + 여유 1시간 — 4일 전량 성공
       · 대상일 중복 시 최신 산출분 채택 → 이력 오염 없음

처리 절차
  ① 사전점검 : 모델·데이터·인증키 상태 확인
  ② 데이터 갱신 : KPX API로 최신 SMP·수요예측 수집(--no-collect 로 생략 가능)
  ③ 데이터셋 재구성 : 전처리 → 통합 → 피처 생성
  ④ 예측 : 익일 24시간 × 육지·제주
  ⑤ 산출 : 예측표(CSV·XLSX)·그래프·실행이력
  ⑥ 사후검증 : 직전 예측일의 실적 대비 오차 자동 산출

설계 원칙
  · 예측 대상일에 실적이 없어도 동작(운영 정상 상태)
  · 단계별 실패 시 사유를 명시하고 후속 단계 진행 여부를 자동 판정
  · 전 실행 이력을 `logs/step6_predict_history.csv` 에 누적 기록
  · **실적 소급 갱신** — 원문 지연·결손 복구분을 기존 이력에 자동 반영(2026.9.7 신설)
  · **자동 폴백 전환** — 수요예측 미공표로 정규 산출 실패 시 21시 이후 축약 모델로 대체(2026.9.10 신설)
  · **Chronos 보조** — 게시 기본=현행 유지. CHRONOS_BLEND_W>0 일 때만 혼성(2026.9.12)
  · **제주 mirror** — 제주 = 육지 최종 예측(편의보정·Chronos 혼성 포함) 복사(2026.10.1 채택)
      근거 : 9월 제주 실적 97.5%가 육지와 동일, 현행 2단계+제주 자체 보정 MAE 13.64 vs 육지값 11.46원
      복귀 : --jeju-mode stage2 또는 환경변수 JEJU_MODE=stage2(2단계 자산 보존)
      육지 산출 실패 시에는 제주 단독(2단계·단일모델) 산출로 자동 대체

버전 : v1.6 (2026-10-01) — 제주 mirror 운영(판정자료 「제주2단계재가동」 v1.0)
       v1.5 (2026-09-14) — 고태양광 연화·9.1패턴 PV가드
실행 : python scripts\\step6_predict.py                  (익일 예측)
       python scripts\\step6_predict.py --date 20260805  (특정일 예측)
       python scripts\\step6_predict.py --no-collect     (수집 생략, 기존 데이터 사용)
       python scripts\\step6_predict.py --dry-run        (사전점검만)
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_api_check import BASE_DIR, ENV_PATH, load_env  # noqa: E402
from step4_baseline import feat_paths, metrics, set_mape_floor, MAPE_FLOOR  # noqa: E402
from step4_train import EXCLUDE, CATEGORICAL  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas·numpy 미설치 → pip install -r requirements.txt")

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predict")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HISTORY = os.path.join(LOG_DIR, "step6_predict_history.csv")

TARGETS = ["smp_land", "smp_jeju"]
STEPS = []          # 실행 이력


def log(단계, 상태, 내용=""):
    STEPS.append({"단계": 단계, "상태": 상태, "내용": 내용})
    mark = {"정상": "○", "주의": "△", "실패": "×", "생략": "-"}.get(상태, " ")
    print(f"  [{mark}] {단계} : {상태}" + (f" — {내용}" if 내용 else ""))


def _dec(b):
    """하위 프로세스 출력 디코딩 — utf-8 우선, 실패 시 cp949(Windows 기본).
    · 콘솔 코드페이지가 949인 환경에서 한글 오류 메시지가 깨지는 문제 대응"""
    if not b:
        return ""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def run(script, args, timeout=1800):
    """하위 스크립트 실행. (성공여부, 마지막 출력) 반환.
    · 실패 시 stderr 요약을 함께 반환하여 원인 진단 가능하도록 구성(v1.1)"""
    cmd = [sys.executable, os.path.join(SCRIPTS, script)] + args
    env = dict(os.environ, PYTHONIOENCODING="utf-8")   # 자식 출력 인코딩 고정
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        out = [x for x in _dec(r.stdout).strip().split("\n") if x.strip()]
        err = [x for x in _dec(r.stderr).strip().split("\n") if x.strip()]
        if r.returncode == 0:
            return True, (out[-1] if out else "")
        # 실패 시 : 예외 유형·메시지가 담긴 stderr 말미를 우선 표시
        msg = " | ".join(err[-3:]) if err else (out[-1] if out else "")
        return False, f"[종료코드 {r.returncode}] {msg[:300]}"
    except subprocess.TimeoutExpired:
        return False, f"타임아웃({timeout}초)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------
# ① 사전점검
# ------------------------------------------------------------------
def precheck(t_list):
    print("\n■ ① 사전점검")
    ok = True

    if os.path.exists(ENV_PATH):
        env = load_env(ENV_PATH)
        keys = [k for k in env if k.endswith("_API_KEY") and env[k]]
        log("인증키", "정상" if keys else "실패", f"{len(keys)}종 확인")
        ok &= bool(keys)
    else:
        log("인증키", "실패", f".env 없음 → {ENV_PATH}")
        ok = False

    for t in t_list:
        mp = os.path.join(MODEL_DIR, f"smp_{t}_lightgbm.pkl")
        alt = os.path.join(MODEL_DIR, f"smp_{t}_sklearn.pkl")
        p = mp if os.path.exists(mp) else (alt if os.path.exists(alt) else None)
        if p:
            age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).days
            log(f"모델({t})", "정상" if age <= 120 else "주의",
                f"학습 후 {age}일 경과" + ("" if age <= 120 else " → 재학습 권고"))
        else:
            log(f"모델({t})", "실패", "모델 파일 없음 → step4_train.py 선행 필요")
            ok = False

        src, _ = feat_paths(t)
        if not os.path.exists(src):
            log(f"피처셋({t})", "실패", "피처셋 없음 → step3_features.py 선행 필요")
            ok = False
            continue
        try:
            d = pd.read_csv(src, encoding="utf-8-sig", usecols=["date"])
            last = d["date"].max()
            gap = (datetime.now().date() - datetime.strptime(last, "%Y-%m-%d").date()).days
            log(f"피처셋({t})", "정상" if gap <= 3 else "주의",
                f"최종일 {last} ({gap}일 경과)")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
            # 파일 잘림·인코딩 손상 → 즉시 중단하되 원인·조치 명시
            log(f"피처셋({t})", "실패",
                f"파일 손상 추정({type(e).__name__}) → "
                f"python scripts\\step3_features.py --target {t} 재실행 필요")
            ok = False
    return ok


# ------------------------------------------------------------------
# ② 데이터 갱신
# ------------------------------------------------------------------
def refresh_data(target_date, collect):
    print("\n■ ② 데이터 갱신")
    if not collect:
        log("API 수집", "생략", "--no-collect 지정")
    else:
        start = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
        # 종료일을 예측 대상일(D+1)까지 지정 — 익일 수요예측 공표분 확보 시도
        #  · 미공표 시 해당 일자만 무데이터 처리되며 후속 단계는 정상 진행(v1.1)
        end = target_date
        okc, msg = run("step1_collect.py",
                       ["--source", "smp", "--start", start, "--end", end], 900)
        log("API 수집", "정상" if okc else "주의",
            f"{start}~{end} / {msg[:60]}")

    # 데이터셋 종료일을 예측 대상일로 명시 — 예측 행 누락 방지(v1.1)
    end_ymd = datetime.strptime(target_date, "%Y%m%d").strftime("%Y%m%d")
    for s, a, nm in (("step2_preprocess.py", ["--source", "smp"], "전처리"),
                     ("step1_collect_vilage_fcst.py", [], "단기예보"),
                     ("step8_pv_forecast.py", ["--apply", "--no-save"], "태양광 전망"),
                     ("step3_build_dataset.py", ["--end", end_ymd], "데이터셋 통합")):
        okr, msg = run(s, a, 900)
        log(nm, "정상" if okr else ("주의" if nm in ("단기예보", "태양광 전망") else "실패"),
            msg[:70])
        if not okr and nm not in ("단기예보", "태양광 전망"):
            return False
    return True


def rebuild_features(t_list):
    print("\n■ ③ 피처 생성")
    okall = True
    for t in t_list:
        okr, msg = run("step3_features.py", ["--target", t], 900)
        log(f"피처({t})", "정상" if okr else "실패", msg[:70])
        okall &= okr
    return okall


# ------------------------------------------------------------------
# ④ 예측
# ------------------------------------------------------------------
def predict(t, target_date, land_pred=None):
    """저장된 모델로 대상일 24시간 예측.
    · land_pred : 육지 예측 결과(제주 2단계 예측 시 사용)
    · 제주는 2단계 모델(ml/models/smp_smp_jeju_stage2.pkl) 보유 시 자동 적용
      예측식 : 제주 = w×육지예측 + (1-w)×(육지예측 + 잔차모델)"""
    import joblib
    stage2 = os.path.join(MODEL_DIR, f"smp_{t}_stage2.pkl")
    use2 = (t == "smp_jeju" and os.path.exists(stage2) and land_pred is not None)

    # VST 앙상블 우선 적용 (2026.9.2 신설, 과제 A-2)
    #  · 4개 성분(vst · vst_l1 · vst_x · diff) 단순평균 — v1.1(2026.9.3) 성분 ② 교체
    #  · 실증 : MAE 8.72 → 7.97원 / rMAE 0.804 → 0.736 (편의보정 포함)
    #  · 성분 ②를 최근성 가중(vst_w) → L1 목적함수(vst_l1)로 교체 — 아침 전환 위상오차 대응
    #  · 제주 2단계 구간은 기존 경로 유지 — 앙상블은 육지·제주 단일 예측에만 적용
    ens_p = os.path.join(MODEL_DIR, f"smp_{t}_ensemble.pkl")
    ens = None
    if not use2 and os.path.exists(ens_p):
        try:
            ens = joblib.load(ens_p)
        except Exception:
            ens = None

    if use2:
        pack = joblib.load(stage2)
    elif ens is not None:
        pack = ens
    else:
        mp = os.path.join(MODEL_DIR, f"smp_{t}_lightgbm.pkl")
        if not os.path.exists(mp):
            mp = os.path.join(MODEL_DIR, f"smp_{t}_sklearn.pkl")
        pack = joblib.load(mp)
    model = pack.get("model")
    feats = pack["features"]

    src, _ = feat_paths(t)
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)

    note = ""
    if ens is not None:
        # 교차시차 24종 부착 — 전일 실적 기반, 23:30 실행 시 확보 가능
        from step3_cross_hour import add_cross_hour
        df, _cr = add_cross_hour(df, t)
        note = f" / 앙상블 {len(ens['members'])}종"
    if use2:
        # 육지 예측값 결합 후 연동 변수 생성(과거 실적 기반 — 예측 시점 확보 가능)
        from step4_jeju_stage2 import add_stage2_features
        lp = land_pred[["date", "hour_n", "예측"]].rename(columns={"예측": "smp_land_pred"})
        df = df.merge(lp, on=["date", "hour_n"], how="left")
        df, _ = add_stage2_features(df)
        note = f" / 2단계(w={pack.get('land_weight', 0):.2f})"

    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")

    d = datetime.strptime(target_date, "%Y%m%d").strftime("%Y-%m-%d")
    seg = df[df["date"] == d]
    if seg.empty:
        return None, f"{d} 행 없음 — 수요예측 미수집 또는 데이터셋 미갱신"
    miss = [c for c in feats if c not in seg.columns]
    if miss:
        return None, f"변수 누락 {len(miss)}종: {miss[:3]}"
    na = seg[feats].isna().sum().sum()

    # ── 입력 건전성 검사 【결함 조치 2026.8.20】 ──────────────
    #  배경 : 본 모델은 수요예측 계열(11종)을 핵심 입력으로 사용하나,
    #         D+1 수요예측은 23시경 공개되므로 그 이전 실행 시 전량 결측됨.
    #         LightGBM 은 결측을 오류 없이 처리하여 **경고 없이 30~40원 낮은 값**을 산출함
    #         (2026.8.18~20 실측 — 예측 110원 vs 실적 148원).
    #  조치 : 핵심 변수 결측 시 산출을 중단하고 실행 시각·대안을 안내
    dem = [c for c in feats if c.startswith("demand_fcst")]
    if dem:
        dem_na = int(seg[dem].isna().all(axis=0).sum())   # 전량 결측인 변수 수
        if dem_na >= max(1, len(dem) // 2):
            return None, (
                f"수요예측 변수 {dem_na}/{len(dem)}종 전량 결측 — 산출 중단\n"
                f"       · 원인 : D+1 공표 시각 변동(2026.9월 실측 14~20시) — 미공표 시각 실행\n"
                f"       · 조치 : 21시 이후 재실행하거나, 주간 예측"
                f"(step6_weekly_predict.py) 사용\n"
                f"       · 주간 예측은 수요예측 미사용 모델(47변수)이므로 시각 무관 산출 가능"
            )

    # 일반 변수 결측률 과다 시에도 중단 — 데이터셋 갱신 누락 등
    na_ratio = float(seg[feats].isna().mean().mean())
    if na_ratio > 0.30:
        return None, (f"입력 결측률 {na_ratio:.1%} 과다 — 산출 중단\n"
                      f"       · 데이터셋 갱신 상태를 확인할 것")

    if use2:
        w = float(pack.get("land_weight", 0.0))
        lpv = seg["smp_land_pred"].to_numpy(float)
        if np.isnan(lpv).all():
            return None, "육지 예측값 미확보 — 육지 예측 선행 필요"
        raw = model.predict(seg[feats])
        base = lpv + raw if pack.get("residual") else raw   # 잔차 학습 시 복원
        p = w * lpv + (1 - w) * base
    elif ens is not None:
        # 앙상블 산출 — 성분별 역변환 후 가중평균
        med = ens["vst"]["median"]
        mad = ens["vst"]["mad"]
        dbase = ens.get("diff_base", "smp_lag_d1")
        ps, wt = [], []
        for mb, wv in zip(ens["members"], ens["weights"]):
            f2 = mb["features"]
            if any(c not in seg.columns for c in f2):
                continue                       # 변수 미비 성분은 제외
            if mb["kind"] == "asinh":
                ps.append(np.sinh(mb["model"].predict(seg[f2])) * mad + med)
            else:
                b = seg[dbase]
                if b.isna().all():
                    continue
                ps.append(mb["model"].predict(seg[f2])
                          + b.fillna(b.mean()).to_numpy(float))
            wt.append(wv)
        if not ps:
            return None, "앙상블 성분 전량 산출 불가 — 변수 확인 필요"
        wt = np.asarray(wt, float)
        p = np.average(np.vstack(ps), axis=0, weights=wt / wt.sum())
        note += f"({len(ps)}/{len(ens['members'])} 성분)"
    else:
        p = model.predict(seg[feats])

    try:
        from step6_wx_pattern import apply_pattern
        rain = seg["rain_daytime"].values if "rain_daytime" in seg.columns else 0
        sol = seg["solar_ratio"].values if "solar_ratio" in seg.columns else 1
        wk = seg["is_weekend"].values if "is_weekend" in seg.columns else 0
        lag = seg["smp_lag_d1"].values if "smp_lag_d1" in seg.columns else None
        pen = (seg["pv_penetration"].values
               if "pv_penetration" in seg.columns else None)
        pv = seg["pv_fcst"].values if "pv_fcst" in seg.columns else None
        dem = (seg["demand_fcst_land"].values
               if "demand_fcst_land" in seg.columns else None)
        p0 = np.asarray(p, dtype=float).copy()
        p = apply_pattern(p, seg["hour_n"].values, rain, sol, wk, lag,
                          pv_pen=pen, pv_fcst=pv, demand_fcst=dem)
        if not np.allclose(p0, p):
            note += " / 9.1패턴"
    except Exception:
        pass

    # 고태양광 → SMP 하향 연화 (2026.9.14 신설 — Duck Curve)
    try:
        from step6_pv_pattern import apply_pv_softening
        lag = seg["smp_lag_d1"].values if "smp_lag_d1" in seg.columns else None
        pen = (seg["pv_penetration"].values
               if "pv_penetration" in seg.columns else None)
        pv = seg["pv_fcst"].values if "pv_fcst" in seg.columns else None
        dem = (seg["demand_fcst_land"].values
               if "demand_fcst_land" in seg.columns else None)
        p, pnote = apply_pv_softening(
            p, seg["hour_n"].values, pv_pen=pen, pv_fcst=pv,
            demand_fcst=dem, lag_d1=lag,
            light_load=(seg["is_light_load_regime"].values
                        if "is_light_load_regime" in seg.columns else None))
        if pnote and "없음" not in pnote and "생략" not in pnote:
            note += f" / {pnote}"
    except Exception as e:
        note += f" / 태양광연화 실패({type(e).__name__})"

    # 시간대별 잔차 편의 보정 (2026.9.2 신설)
    #  · 트리 모델의 평균회귀 축소로 수준 급변 구간에서 계통적 과소예측 발생
    #  · 직전 21일 운영 이력의 시간대별 잔차를 γ 감쇠 적용하여 상쇄(2026.9.7 γ=1.5)
    #  · 실증 : 육지 −0.21%p(급등구간 −1.13%p) / 제주 −0.73%p(−1.80%p)
    try:
        from step6_bias_adjust import apply_bias
        p, bnote = apply_bias(p, seg["hour_n"].values, t,
                              seg["date"].iloc[0])
        note += f" / {bnote}"
    except Exception as e:
        note += f" / 편의보정 실패({type(e).__name__})"

    out = seg[["date", "hour_n"]].copy()
    out["예측"] = np.round(p, 2)
    out["실적"] = seg[t].values if t in seg.columns else np.nan
    return out, f"{len(out)}시간 / 입력결측 {int(na)}건{note}"


# ------------------------------------------------------------------
# 실적 소급 갱신 (v1.2, 2026.9.7 신설)
# ------------------------------------------------------------------
#  · 배경 : 예측 이력의 '실적'은 산출 시점 값으로 1회 기록되어 이후 갱신되지 않음
#           → 원문 지연·결손(제주 8일) 복구 후에도 공란이 잔존
#  · 영향 : 편의 보정의 잔차 산정 이력에서 해당일 제외 — 보정 정밀도 저하
#           (2026.9.7 실측 : 육지 평일 19일 대비 제주 15일로 축소)
#  · 조치 : 저장 직전 최신 피처셋과 대조하여 **결측 실적만** 소급 보완
def backfill_actual(old, t):
    """기존 이력의 실적 결측분을 최신 피처셋 값으로 보완 → (이력, 보완건수)"""
    if old.empty or "실적" not in old.columns:
        return old, 0
    try:
        src, _ = feat_paths(t)
        a = pd.read_csv(src, encoding="utf-8-sig",
                        usecols=["date", "hour_n", t])
    except Exception:
        return old, 0
    a = a.rename(columns={t: "_실적"}).drop_duplicates(["date", "hour_n"])
    m = old.merge(a, on=["date", "hour_n"], how="left")
    na = m["실적"].isna()
    n = int((na & m["_실적"].notna()).sum())
    m["실적"] = m["실적"].fillna(m["_실적"])
    return m.drop(columns=["_실적"]), n


# ------------------------------------------------------------------
# ⑥ 사후검증
# ------------------------------------------------------------------
def verify_past(t):
    """직전 예측 이력 중 실적이 확보된 건의 오차 산출."""
    hist = os.path.join(OUT_DIR, f"predictions_{t}.csv")
    if not os.path.exists(hist):
        return None
    h = pd.read_csv(hist, encoding="utf-8-sig")
    src, _ = feat_paths(t)
    a = pd.read_csv(src, encoding="utf-8-sig", usecols=["date", "hour_n", t])
    m = h.merge(a, on=["date", "hour_n"], how="inner", suffixes=("", "_실적"))
    m = m.dropna(subset=[t])
    if m.empty:
        return None
    return m, metrics(m[t], m["예측"])


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="step6 운영 예측")
    ap.add_argument("--date", default=None, help="예측 대상일 YYYYMMDD(기본 익일)")
    ap.add_argument("--target", default="all", choices=TARGETS + ["all"])
    ap.add_argument("--no-collect", action="store_true", help="API 수집 생략")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="전처리·피처 재생성 생략(피처셋이 최신인 경우)")
    ap.add_argument("--dry-run", action="store_true", help="사전점검만 수행")
    ap.add_argument("--backfill-only", action="store_true",
                    help="예측 없이 예측 이력의 실적 결측만 최신 피처셋으로 소급(2026.10.2 신설)")
    ap.add_argument("--mape-floor", type=float, default=MAPE_FLOOR)
    ap.add_argument("--jeju-mode", default=os.environ.get("JEJU_MODE", "mirror") or "mirror",
                    choices=["mirror", "stage2"],
                    help="제주 산출 방식 — mirror(기본, 육지 최종 예측 복사)·stage2(2단계 모델, 복귀용)")
    ap.add_argument("--fallback", default="auto", choices=["off", "auto", "force"],
                    help="정규 산출 실패 시 폴백(수요예측 미사용 47변수 축약 모델) 전환 — "
                         "auto(기본) : 21시 이후 실패 시 자동 / force : 시각 무관 / off : 미사용")
    args = ap.parse_args()
    set_mape_floor(args.mape_floor)

    t_list = TARGETS if args.target == "all" else [args.target]
    target_date = args.date or (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    d_disp = datetime.strptime(target_date, "%Y%m%d").strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = datetime.now()

    print("=" * 78)
    print(" step6 운영 예측 — 익일 시간별 SMP (v1.6)")
    print(f" 실행시각 : {t0:%Y-%m-%d %H:%M:%S}")
    print(f" 예측대상 : {d_disp} / 계통 : {', '.join(t_list)}")
    print("=" * 78)

    if args.backfill_only:
        #  실적 지연 공표·웹 보완 후 즉시 반영용 — 예측 산출물·실행 이력은 건드리지 않음
        print("\n■ 실적 소급 갱신 (--backfill-only)")
        for t in t_list:
            hist = os.path.join(OUT_DIR, f"predictions_{t}.csv")
            if not os.path.exists(hist):
                log(f"실적 소급 갱신({t})", "생략", "이력 없음")
                continue
            old = pd.read_csv(hist, encoding="utf-8-sig")
            old, nfill = backfill_actual(old, t)
            if nfill:
                old.to_csv(hist, index=False, encoding="utf-8-sig")
            log(f"실적 소급 갱신({t})", "정상", f"{nfill}건 보완")
        #  일일 예측표(SMP예측_YYYYMMDD.csv)의 실적 열 — 통합 대시보드 표시용(2026.10.2 보완)
        #  · 최근 14일 파일 중 실적이 빈 시간만 채움(예측값·산출구분 열은 불변)
        acts = {}
        for t in t_list:
            try:
                src, _ = feat_paths(t)
                a = pd.read_csv(src, encoding="utf-8-sig", usecols=["date", "hour_n", t])
                acts[t] = a.dropna(subset=[t]).drop_duplicates(["date", "hour_n"])
            except Exception:
                continue
        cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y%m%d")
        nd = 0
        for f in sorted(glob.glob(os.path.join(OUT_DIR, "SMP예측_[0-9]*.csv"))):
            ymd = os.path.basename(f)[6:14]
            if not ymd.isdigit() or ymd < cutoff:
                continue
            d = pd.read_csv(f, encoding="utf-8-sig")
            n0 = 0
            for t, a in acts.items():
                col = f"실적_{t}"
                if col not in d.columns:
                    continue
                m = d[["date", "hour_n"]].merge(a, on=["date", "hour_n"], how="left")[t]
                na = d[col].isna() & m.notna()
                n0 += int(na.sum())
                d.loc[na, col] = m[na].values
            if n0:
                d.to_csv(f, index=False, encoding="utf-8-sig")
                log(f"일일 예측표 실적 보완", "정상", f"{os.path.basename(f)} {n0}건")
                nd += 1
        if not nd:
            log("일일 예측표 실적 보완", "정상", "보완 대상 없음")
        return

    ok = precheck(t_list)
    if args.dry_run:
        print("\n[종료] --dry-run 지정 — 사전점검만 수행")
        return
    if not ok:
        print("\n[중단] 사전점검 실패 → 상기 조치 후 재실행")
        sys.exit(1)

    if args.no_rebuild:
        print("\n■ ② 데이터 갱신")
        log("전처리·피처", "생략", "--no-rebuild 지정 — 기존 피처셋 사용")
    else:
        if not refresh_data(target_date, not args.no_collect):
            print("\n[중단] 데이터 갱신 실패")
            sys.exit(1)
        if not rebuild_features(t_list):
            print("\n[중단] 피처 생성 실패")
            sys.exit(1)

    # ── ④ 예측 ─────────────────────────────────
    print("\n■ ④ 예측 산출")
    results = {}
    # 제주 2단계 예측이 육지 결과를 입력으로 사용하므로 육지를 선행 처리
    order = sorted(t_list, key=lambda x: 0 if x == "smp_land" else 1)
    mirror = (args.jeju_mode == "mirror" and "smp_jeju" in t_list
              and "smp_land" in t_list)
    for t in order:
        if t == "smp_jeju" and mirror and "smp_land" in results:
            continue                     # 제주는 Chronos 혼성 후 육지 최종값 복사(④-2)
        out, msg = predict(t, target_date, land_pred=results.get("smp_land"))
        if out is None:
            log(f"예측({t})", "실패", msg)
            continue
        log(f"예측({t})", "정상", msg)
        results[t] = out
        v = out["예측"]
        print(f"      평균 {v.mean():.1f} / 최고 {v.max():.1f}({int(out.loc[v.idxmax(), 'hour_n'])}시)"
              f" / 최저 {v.min():.1f}({int(out.loc[v.idxmin(), 'hour_n'])}시) 원/kWh")

    # ── ④-1 Chronos 보조(shadow) · 혼성 (2026.9.12 시험, 9.27 운영 채택)
    #  · W=0.30 · START=9.14 · END 미지정(무기한)
    #  · torch 부재 시 기 산출 shadow 파일로 혼성 가능
    if results:
        try:
            from step8_chronos_shadow import (
                produce, apply_blend, load_shadow_map, blend_weight)
            targets_ch = [t for t in results if t == "smp_land"] or list(
                results.keys())[:1]
            cmap = {}
            src = "live"
            try:
                cmap = produce(d_disp, targets=targets_ch)
            except Exception as e_prod:
                cmap = load_shadow_map(d_disp, targets_ch)
                src = f"file({type(e_prod).__name__})"
            if not cmap:
                log("Chronos 보조", "생략",
                    "shadow 없음 — chronos 설치 또는 step8_chronos_shadow 선행")
            else:
                applied, note = apply_blend(results, cmap, day=d_disp)
                w = blend_weight(d_disp)
                if applied:
                    log("Chronos 혼성", "정상", f"{note} / src={src}")
                else:
                    log("Chronos 보조", "정상",
                        f"{note} / src={src} / w_eff={w}")
        except ImportError as e:
            log("Chronos 보조", "생략", str(e)[:120])
        except Exception as e:
            log("Chronos 보조", "주의", f"{type(e).__name__}: {str(e)[:100]}")

    # ── ④-2 제주 mirror (2026.10.1 채택) ─────────────────────────────
    #  · 제주 = 육지 최종 예측(편의보정·Chronos 혼성 반영 후) — 실적은 제주 피처셋 값
    #  · 육지 실패 시 위 루프에서 제주가 단독 산출되므로 여기서는 미적용
    if mirror and "smp_land" in results and "smp_jeju" not in results:
        jo = results["smp_land"][["date", "hour_n", "예측"]].copy()
        try:
            src_j, _ = feat_paths("smp_jeju")
            aj = pd.read_csv(src_j, encoding="utf-8-sig",
                             usecols=["date", "hour_n", "smp_jeju"])
            jo = jo.merge(aj.rename(columns={"smp_jeju": "실적"}),
                          on=["date", "hour_n"], how="left")
        except Exception:
            jo["실적"] = np.nan
        results["smp_jeju"] = jo
        v = jo["예측"]
        log("예측(smp_jeju)", "정상",
            f"{len(jo)}시간 / mirror — 육지 최종 예측 복사(편의보정·혼성 포함)")
        print(f"      평균 {v.mean():.1f} / 최고 {v.max():.1f}({int(jo.loc[v.idxmax(), 'hour_n'])}시)"
              f" / 최저 {v.min():.1f}({int(jo.loc[v.idxmin(), 'hour_n'])}시) 원/kWh")
    elif mirror and "smp_jeju" in results:
        log("제주 mirror", "주의", "육지 산출 실패 — 제주 단독 산출로 대체")

    if not results:
        #  ── A-9 자동 폴백 전환 (2026.9.10 신설) ────────────────────────
        #   배경 : 2026.9.9 대상 산출이 18:00·21:00·23:30 3단 전량 실패
        #          (D+1 수요예측 미공표 — 종전 실측 경계 14~20시를 초과)
        #   구조 : 3단은 실행 시각만 다른 **동일 모델**이므로 진정한 폴백이 아님
        #   조치 : 실패 시 수요예측 미사용 47변수 축약 모델(주간 예측 D+1)로 전환
        #   전환 조건(auto) : 21시 이후 — 그 이전 실패는 '공표 대기'로 정상 상태
        hour_now = datetime.now().hour
        do_fb = (args.fallback == "force"
                 or (args.fallback == "auto" and hour_now >= 21))
        if do_fb:
            print("\n■ ④-2 폴백 전환 — 수요예측 미사용 축약 모델")
            log("정규 산출", "실패", "수요예측 미공표 — 폴백 전환")
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from step6_fallback_predict import (latest_weekly, KIND_NOTE)
                import subprocess
                r = subprocess.run(
                    [sys.executable,
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "step6_fallback_predict.py"),
                     "--date", target_date, "--force"],
                    capture_output=True, text=True, timeout=300,
                    env={**os.environ, "JEJU_MODE": args.jeju_mode})
                print(r.stdout[-2000:] if r.stdout else "")
                if r.returncode == 0:
                    log("폴백 산출", "정상", KIND_NOTE)
                    print("\n[완료] 폴백 산출로 대체 — 공표 확인 후 재실행 권고")
                    print("=" * 78)
                    return
                log("폴백 산출", "실패", (r.stderr or "")[-200:])
            except Exception as e:
                log("폴백 산출", "실패", f"{type(e).__name__}")
        else:
            log("폴백 전환", "미적용",
                f"현재 {hour_now}시 — 21시 이전 실패는 공표 대기(정상)"
                if args.fallback == "auto" else "--fallback off 지정")
        print("\n[중단] 예측 산출 실패")
        sys.exit(1)

    # ── ⑤ 산출물 ───────────────────────────────
    print("\n■ ⑤ 산출물 저장")
    merged = None
    for t, out in results.items():
        o = out.rename(columns={"예측": f"예측_{t}", "실적": f"실적_{t}"})
        merged = o if merged is None else merged.merge(o, on=["date", "hour_n"], how="outer")

        hist = os.path.join(OUT_DIR, f"predictions_{t}.csv")
        rec = out.copy()
        rec.insert(0, "예측생성시각", f"{t0:%Y-%m-%d %H:%M:%S}")
        if os.path.exists(hist):
            old = pd.read_csv(hist, encoding="utf-8-sig")
            old = old[~((old["date"] == rec["date"].iloc[0]))]
            old, nfill = backfill_actual(old, t)      # 실적 소급 갱신(v1.2)
            if nfill:
                log(f"실적 소급 갱신({t})", "정상", f"{nfill}건 보완")
            rec = pd.concat([old, rec], ignore_index=True)
        rec.to_csv(hist, index=False, encoding="utf-8-sig")

    daily = os.path.join(OUT_DIR, f"SMP예측_{target_date}.csv")
    merged.to_csv(daily, index=False, encoding="utf-8-sig")
    log("일일 예측표", "정상", daily)

    try:
        xl = os.path.join(OUT_DIR, f"SMP예측_{target_date}.xlsx")
        with pd.ExcelWriter(xl, engine="xlsxwriter") as w:
            merged.to_excel(w, sheet_name="익일예측", index=False)
            pd.DataFrame(STEPS).to_excel(w, sheet_name="실행이력", index=False)
        log("Excel 산출", "정상", xl)
    except Exception as e:
        log("Excel 산출", "주의", f"{type(e).__name__} — CSV로 대체")

    # 그래프
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        ko = False
        for f in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
            if f in have:
                plt.rcParams["font.family"] = f
                ko = True
                break
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(11, 4.6))
        for t, col in zip(results, ["#1F3864", "#C00000"]):
            lb = ("육지" if t == "smp_land" else "제주") if ko else t
            ax.plot(results[t]["hour_n"], results[t]["예측"], marker="o", ms=4,
                    lw=2, label=lb, color=col)
        ax.set_title((f"익일 시간별 SMP 예측 — {d_disp}" if ko
                      else f"Hourly SMP forecast — {d_disp}"), fontsize=13, weight="bold")
        ax.set_xlabel("거래시간" if ko else "Hour")
        ax.set_ylabel("원/kWh" if ko else "KRW/kWh")
        ax.set_xticks(range(1, 25)); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        gp = os.path.join(OUT_DIR, f"SMP예측_{target_date}.png")
        fig.savefig(gp, dpi=130); plt.close(fig)
        log("그래프", "정상", gp)
    except Exception as e:
        log("그래프", "주의", type(e).__name__)

    # ── ⑥ 사후검증 ─────────────────────────────
    print("\n■ ⑥ 사후검증 (과거 예측 대비 실적)")
    for t in t_list:
        v = verify_past(t)
        if v is None:
            log(f"검증({t})", "생략", "실적 확보분 없음")
            continue
        m, mm = v
        log(f"검증({t})", "정상",
            f"{m['date'].nunique()}일 {len(m)}시간 / MAPE {mm['MAPE']:.2f}% "
            f"/ RMSE {mm['RMSE']:.2f}")

    # ── 이력 기록 ──────────────────────────────
    el = (datetime.now() - t0).total_seconds()
    row = {"실행시각": f"{t0:%Y-%m-%d %H:%M:%S}", "예측대상일": d_disp,
           "계통": ",".join(results.keys()), "소요(초)": round(el, 1),
           "실패단계": ";".join(s["단계"] for s in STEPS if s["상태"] == "실패") or "-",
           "주의단계": ";".join(s["단계"] for s in STEPS if s["상태"] == "주의") or "-"}
    for t, out in results.items():
        row[f"평균_{t}"] = round(float(out["예측"].mean()), 2)
    h = pd.DataFrame([row])
    if os.path.exists(HISTORY):
        h = pd.concat([pd.read_csv(HISTORY, encoding="utf-8-sig"), h], ignore_index=True)
    h.to_csv(HISTORY, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f" 예측 완료 : {d_disp} / 소요 {el:.1f}초")
    for t, out in results.items():
        nm = "육지" if t == "smp_land" else "제주"
        print(f"   {nm} 일평균 {out['예측'].mean():.1f}원/kWh "
              f"(최고 {out['예측'].max():.1f} / 최저 {out['예측'].min():.1f})")
    print(f" 산출물 : {OUT_DIR}")
    print(f" 실행이력 : {HISTORY}")
    print("=" * 78)


if __name__ == "__main__":
    main()
