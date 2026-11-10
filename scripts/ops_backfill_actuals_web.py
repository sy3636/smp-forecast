# -*- coding: utf-8 -*-
"""
ops_backfill_actuals_web.py — API 공표 지연일 실적 일괄 반영 (v1.1, 수동 실행)
==============================================================================
배경
  · 2026. 10. 2.~ 공공데이터포털 API 0건 지속 — 실적은 매일 웹 보완 수집 후 5단계 수동 반영
  · 운영매뉴얼 Ⅵ-V 절차를 1회 실행으로 묶음(대상일 인자·원문 미확보 시 중단·검증 출력)

동작
  ① step1_collect_kpx_web.py --dates — 웹 보완 수집(API 원문 없는 날만 저장, 대조 불일치 시 수집기 중단)
  ② 대상일 원문(API 또는 웹) 확인 — 없으면 이후 단계 미실행
  ③ step2_preprocess → step3_build_dataset → step3_features(육지·제주)
  ④ step6_predict.py --backfill-only --no-collect — 예측 이력·일일 예측표 실적 소급
  ④-1 step6_weekly_hist.py — 주간 예측 이력 실적 재결합(15:00 정규 경로와 동일)
  ④-2 Chronos 파일 실적 결측 행 소급(피처 확정값, 예측·생성시각 불변, .bak_YYYYMMDD 백업)
  ⑤ step11_warehouse_load.py — Grafana 반영
  ⑥ 검증 — 이력·피처 실적 결측, 웨어하우스 모델별 실적 건수

유의
  · Chronos 파일은 산출 시점 실적만 기록, 주간은 15:00에만 재결합 — API 지연일은 ④-1·2 없이는 Grafana 공란(10. 5. 사례)
  · 로컬 저장소 파일을 stdin으로 전달 — 이미지 재빌드 없이 실행 가능

실행 (로컬 PC → 서버, stdin 전달)
  ssh smp 'docker exec -i -w /app smp-forecast python - 20261006' < scripts/ops_backfill_actuals_web.py
  ssh smp 'docker exec -i -w /app smp-forecast python - 20261006,20261007' < scripts/ops_backfill_actuals_web.py
  ssh smp 'docker exec -i -w /app smp-forecast python - --dry-run 20261006' < scripts/ops_backfill_actuals_web.py

인자
  날짜      YYYYMMDD — 쉼표 또는 공백 구분 복수 가능, 생략 시 오늘(KST)
  --dry-run 웹 조회·대조만(저장·반영 없음)

버전 : v1.1 (2026-10-05) — ④-1 주간 재결합·④-2 Chronos 실적 소급 추가
       v1.0 (2026-10-05) — 최초
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

os.chdir("/app")
sys.path.insert(0, "/app/scripts")

KST = timezone(timedelta(hours=9))


def parse_args(argv):
    dry = "--dry-run" in argv
    raw = [t for a in argv if a != "--dry-run" for t in a.replace(",", " ").split()]
    if not raw:
        raw = [datetime.now(KST).strftime("%Y%m%d")]
    days = []
    for t in raw:
        try:
            days.append(datetime.strptime(t, "%Y%m%d").strftime("%Y%m%d"))
        except ValueError:
            raise SystemExit(f"[오류] 날짜 형식 YYYYMMDD 아님 : {t}")
    return sorted(set(days)), dry


def run(cmd, env=None):
    print("\n>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd="/app", env=env)
    if r.returncode != 0:
        raise SystemExit(f"fail: {cmd} -> {r.returncode}")


def has_raw(d):
    return os.path.exists(f"data/raw/smp/{d}.json") or os.path.exists(f"data/raw/smp_web/{d}.json")


days, dry = parse_args(sys.argv[1:])
iso = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in days]
print(f"대상일 : {', '.join(iso)}{' / DRY-RUN' if dry else ''}")

# 1) 웹 보완 수집 (API 원문 없는 날만 저장 — 수집기 자체 판단)
collect = [sys.executable, "scripts/step1_collect_kpx_web.py", "--dates", ",".join(days)]
if dry:
    run(collect + ["--dry-run"])
    raise SystemExit("\nDRY-RUN 종료 — 저장·반영 없음")
run(collect)

missing = [d for d in days if not has_raw(d)]
if missing:
    raise SystemExit(f"\n[중단] 원문 미확보 : {', '.join(missing)} — 홈페이지 공표 후 재실행")

# 2) 전처리·데이터셋·피처 (실적 반영)
run([sys.executable, "scripts/step2_preprocess.py", "--source", "smp"])
run([sys.executable, "scripts/step3_build_dataset.py"])
run([sys.executable, "scripts/step3_features.py", "--target", "smp_land"])
run([sys.executable, "scripts/step3_features.py", "--target", "smp_jeju"])

# 3) 예측 이력·일일표 실적 소급
run([sys.executable, "scripts/step6_predict.py", "--backfill-only", "--no-collect"])

# 3-1) 주간 실적 재결합 — 15:00 주간 예측 직후 호출되는 정규 경로(step6_weekly_hist)
run([sys.executable, "scripts/step6_weekly_hist.py"])

# 3-2) Chronos 실적 소급 — 산출 시점 실적만 기록(사후 보완 경로 없음)
#      예측·생성시각은 그대로, 실적 결측 행만 피처(smp_land) 확정값으로 채움
import pandas as pd

feat = pd.read_csv("data/processed/smp_features_smp_land.csv", encoding="utf-8-sig",
                   usecols=["date", "hour_n", "smp_land"])
feat["date"] = feat["date"].astype(str).str[:10]
act = feat.dropna(subset=["smp_land"]).set_index(["date", "hour_n"])["smp_land"]
stamp = datetime.now(KST).strftime("%Y%m%d")

print("\n■ Chronos 실적 소급")
for rel, dcol, hcol in [("outputs/predict/chronos_shadow_smp_land.csv", "date", "hour_n")]:
    if not os.path.exists(rel):
        print(f"  · {rel} : 없음 — 생략")
        continue
    df = pd.read_csv(rel, encoding="utf-8-sig")
    key = list(zip(df[dcol].astype(str).str[:10], pd.to_numeric(df[hcol], errors="coerce")))
    fill = pd.Series([act.get(k) for k in key], index=df.index, dtype="float64")
    m = df["실적"].isna() & fill.notna()
    if not m.any():
        print(f"  · {rel} : 보완 대상 없음")
        continue
    bak = f"{rel}.bak_{stamp}"
    if not os.path.exists(bak):
        subprocess.run(["cp", "-p", rel, bak], check=True)
    df.loc[m, "실적"] = fill[m]
    df.to_csv(rel, index=False, encoding="utf-8-sig")
    by_day = df.loc[m, dcol].astype(str).str[:10].value_counts().sort_index()
    print(f"  · {rel} : {int(m.sum())}건 보완 ({', '.join(f'{d} {n}' for d, n in by_day.items())}) / 백업 {bak}")

# 4) 웨어하우스 재적재
env = os.environ.copy()
env.setdefault("PGHOST", "postgres")
run([sys.executable, "scripts/step11_warehouse_load.py"], env=env)

# 5) 검증
print("\n=== history after (실적 결측 / 평균) ===")
for t in ("smp_land", "smp_jeju"):
    h = pd.read_csv(f"outputs/predict/predictions_{t}.csv", encoding="utf-8-sig")
    h["date"] = h["date"].astype(str).str[:10]
    for d in iso:
        x = h[h["date"] == d]
        print(t, d, "rows", len(x), "actual_na", int(x["실적"].isna().sum()) if len(x) else -1,
              "mean", round(float(x["실적"].mean()), 2) if len(x) and x["실적"].notna().any() else None)

print("\n=== features after ===")
for t in ("smp_land", "smp_jeju"):
    f = pd.read_csv(f"data/processed/smp_features_{t}.csv", encoding="utf-8-sig", usecols=["date", t])
    f["date"] = f["date"].astype(str).str[:10]
    for d in iso:
        x = f[f["date"] == d]
        print(t, d, "smp_na", int(x[t].isna().sum()),
              "mean", round(float(x[t].mean()), 2) if x[t].notna().any() else None)

print("\n=== warehouse after (D+1, 모델별 실적 건수) ===")
os.environ.setdefault("PGHOST", "postgres")
import psycopg
from step11_warehouse_load import dsn

with psycopg.connect(dsn()) as conn:
    rows = conn.execute(
        """SELECT target_date, region, model, COUNT(*), COUNT(actual),
                  ROUND(AVG(pred)::numeric, 1), ROUND(AVG(actual)::numeric, 1)
           FROM smp.fact_hourly_forecast
           WHERE horizon = 'D+1' AND target_date = ANY(%s::date[])
           GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""", (iso,)).fetchall()
for r in rows:
    print(*r)
print("※ weekly 는 D+1 행만 표시 — D+2~ 지평도 동일 파일에서 소급됨")

print("\nFIX_DONE")
