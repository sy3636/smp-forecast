# -*- coding: utf-8 -*-
"""
step10_manual_adjust.py — 현업 수동 보정 엔진 (v1.0)
==============================================================================
목적
  · 협업 부서(현업)가 모델 예측 결과를 **직접 보정**할 수 있는 기능 제공
  · 대상 : ① 시간별 예측(D+1) ② 주간 예측(D+1~D+7) ③ 월별 전망

설계 원칙 (2026. 9. 8. 확정)
  ① **원본 보존** — 모델 예측은 불변. 보정본을 **별도 열·별도 파일**로 관리
     · 성능평가·편의보정은 **원본 기준 유지** → 모델 성능 지표의 오염 방지
     · 대외 제공·업무 활용은 보정본 사용
  ② **입력 방식 2종 병행**
     · 구간 일괄 : 시간대 구간에 ±원 또는 ±% 가감
     · 시간대별 : 개별 시점 직접 지정
  ③ **즉시 반영 + 이력 기록** — 담당자·시각·사유·변경 전후를 전량 보존

파일 구조
  outputs/adjust/
    ├ adjust_rules_{kind}.csv   현재 유효 보정 규칙(대상별 1행 = 1규칙)
    ├ adjust_applied_{kind}.csv 보정 결과(원본·보정량·보정본)
    └ adjust_history.csv        전 조작 이력(추가·삭제·초기화)

  kind : hourly(시간별) · weekly(주간) · monthly(월별)

보정 규칙 열
  대상키   : hourly=대상일(YYYY-MM-DD) / weekly=기준일 / monthly=대상월(YYYY-MM)
  계통     : smp_land · smp_jeju · (월별은 종합)
  범위     : 시간대 구간 "10-16" 또는 개별 "13" / 월별은 "평일"·"휴일"·"종합"
  방식     : add(원) · pct(%) · set(지정값)
  값       : 수치
  사유     : 자유 기재(필수)
  담당자   : 입력자

안전장치
  · 보정폭 상한 : 절대 ±50원 또는 ±30% (초과 시 거부) — 오입력 방지
  · 원본 파일 미변경 — 보정본은 접미사 `_보정` 파일로 별도 저장
  · 규칙 삭제·초기화도 이력에 기록

실행
  python scripts\\step10_manual_adjust.py --list --kind hourly
  python scripts\\step10_manual_adjust.py --add --kind hourly --key 2026-09-09 ^
         --target smp_land --range 10-16 --mode add --value -8 ^
         --reason "현업 판단 — 주간 태양광 유입" --user 홍길동
  python scripts\\step10_manual_adjust.py --apply --kind hourly --key 2026-09-09
  python scripts\\step10_manual_adjust.py --delete --kind hourly --rule-id 3 --user 홍길동

버전 : v1.0 (2026-09-08)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADJ_DIR = os.path.join(BASE_DIR, "outputs", "adjust")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predict")
FCST_DIR = os.path.join(BASE_DIR, "outputs", "forecast")

KINDS = ("hourly", "weekly", "monthly")
MODES = ("add", "pct", "set")
MAX_ABS, MAX_PCT = 50.0, 30.0        # 오입력 방지 상한
RULE_COLS = ["규칙ID", "대상키", "계통", "범위", "방식", "값",
             "사유", "담당자", "등록시각", "상태"]
HIST_COLS = ["시각", "담당자", "조작", "구분", "대상키", "계통",
             "범위", "방식", "값", "사유", "비고"]


# ------------------------------------------------------------------
# 공통
# ------------------------------------------------------------------
def rules_path(kind):
    return os.path.join(ADJ_DIR, f"adjust_rules_{kind}.csv")


def applied_path(kind):
    return os.path.join(ADJ_DIR, f"adjust_applied_{kind}.csv")


def hist_path():
    return os.path.join(ADJ_DIR, "adjust_history.csv")


def load_rules(kind):
    p = rules_path(kind)
    if not os.path.exists(p):
        return pd.DataFrame(columns=RULE_COLS)
    d = pd.read_csv(p, encoding="utf-8-sig")
    for c in RULE_COLS:
        if c not in d.columns:
            d[c] = ""
    return d[RULE_COLS]


def save_rules(kind, d):
    os.makedirs(ADJ_DIR, exist_ok=True)
    d.to_csv(rules_path(kind), index=False, encoding="utf-8-sig")


def log_history(row):
    os.makedirs(ADJ_DIR, exist_ok=True)
    p = hist_path()
    d = pd.DataFrame([row], columns=HIST_COLS)
    d.to_csv(p, mode="a", header=not os.path.exists(p),
             index=False, encoding="utf-8-sig")


def parse_range(rng, kind):
    """범위 문자열 → 시간대 목록. 월별은 구분 문자열 그대로 반환"""
    s = str(rng).strip()
    if kind == "monthly":
        return s if s in ("평일", "휴일", "종합") else "종합"
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def validate(mode, value):
    """오입력 방지 — 상한 초과 시 사유 반환"""
    if mode not in MODES:
        return f"방식 오류({mode}) — add·pct·set 중 지정"
    v = float(value)
    if mode == "add" and abs(v) > MAX_ABS:
        return f"보정폭 상한 초과 — |{v:.1f}| > {MAX_ABS:.0f}원"
    if mode == "pct" and abs(v) > MAX_PCT:
        return f"보정률 상한 초과 — |{v:.1f}| > {MAX_PCT:.0f}%"
    if mode == "set" and not (0 <= v <= 1000):
        return f"지정값 범위 이탈 — {v:.1f}원 (0~1000)"
    return ""


def apply_value(base, mode, v):
    if mode == "add":
        return base + v
    if mode == "pct":
        return base * (1.0 + v / 100.0)
    return np.full_like(np.asarray(base, float), float(v))


# ------------------------------------------------------------------
# 규칙 조작
# ------------------------------------------------------------------
def add_rule(kind, key, target, rng, mode, value, reason, user):
    if kind not in KINDS:
        sys.exit(f"[중단] 구분 오류 — {KINDS}")
    if not str(reason).strip():
        sys.exit("[중단] 사유는 필수 입력")
    #  담당자 필수 — 이력 추적성 확보(2026.9.8 보완)
    if not str(user).strip() or str(user).strip() == "미기재":
        sys.exit("[중단] 담당자는 필수 입력 — 이력 추적을 위해 성명 기재")
    err = validate(mode, value)
    if err:
        sys.exit(f"[중단] {err}")
    d = load_rules(kind)
    rid = (int(d["규칙ID"].max()) + 1) if len(d) else 1
    row = {"규칙ID": rid, "대상키": key, "계통": target, "범위": rng,
           "방식": mode, "값": float(value), "사유": reason,
           "담당자": user, "등록시각": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
           "상태": "유효"}
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    save_rules(kind, d)
    log_history({"시각": row["등록시각"], "담당자": user, "조작": "추가",
                 "구분": kind, "대상키": key, "계통": target, "범위": rng,
                 "방식": mode, "값": value, "사유": reason,
                 "비고": f"규칙ID {rid}"})
    print(f"[등록] 규칙ID {rid} — {kind} / {key} / {target} / {rng} / "
          f"{mode} {value}")
    return rid


def delete_rule(kind, rule_id, user, reason="현업 요청"):
    d = load_rules(kind)
    m = d["규칙ID"].astype(str) == str(rule_id)
    if not m.any():
        sys.exit(f"[중단] 규칙ID {rule_id} 없음")
    r = d[m].iloc[0]
    d.loc[m, "상태"] = "삭제"
    save_rules(kind, d)
    log_history({"시각": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "담당자": user,
                 "조작": "삭제", "구분": kind, "대상키": r["대상키"],
                 "계통": r["계통"], "범위": r["범위"], "방식": r["방식"],
                 "값": r["값"], "사유": reason, "비고": f"규칙ID {rule_id}"})
    print(f"[삭제] 규칙ID {rule_id}")


def list_rules(kind, key=None):
    d = load_rules(kind)
    d = d[d["상태"] == "유효"]
    if key:
        d = d[d["대상키"].astype(str) == str(key)]
    if d.empty:
        print(f"  · 유효 보정 규칙 없음 ({kind}"
              + (f" / {key}" if key else "") + ")")
        return d
    print(d.to_string(index=False))
    return d


# ------------------------------------------------------------------
# 보정 적용
# ------------------------------------------------------------------
def _src_hourly(key):
    p = os.path.join(PRED_DIR, f"SMP예측_{str(key).replace('-', '')}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 산출물 없음 → {p}")
    d = pd.read_csv(p, encoding="utf-8-sig")
    d.columns = [c.lstrip("﻿") for c in d.columns]
    return d, p


def _src_weekly(key):
    p = os.path.join(PRED_DIR, f"SMP주간예측_{str(key).replace('-', '')}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 산출물 없음 → {p}")
    d = pd.read_csv(p, encoding="utf-8-sig")
    d.columns = [c.lstrip("﻿") for c in d.columns]
    return d, p


def _src_monthly(key):
    p = os.path.join(FCST_DIR, f"SMP월별전망_{str(key).replace('-', '')}.csv")
    if not os.path.exists(p):
        sys.exit(f"[중단] 산출물 없음 → {p}")
    raw = open(p, encoding="utf-8-sig").read().splitlines()
    head = [ln for ln in raw if ln.startswith("#") or ln.startswith("﻿#")]
    body = [ln for ln in raw if not (ln.startswith("#") or ln.startswith("﻿#"))]
    from io import StringIO
    d = pd.read_csv(StringIO("\n".join(body)))
    d.columns = [c.lstrip("﻿") for c in d.columns]
    return d, p, head


def apply_rules(kind, key, quiet=False):
    """유효 규칙을 적용하여 보정본 산출 — 원본 파일은 미변경"""
    rules = load_rules(kind)
    rules = rules[(rules["상태"] == "유효")
                  & (rules["대상키"].astype(str) == str(key))]
    head = []
    if kind == "hourly":
        d, src = _src_hourly(key)
    elif kind == "weekly":
        d, src = _src_weekly(key)
    else:
        d, src, head = _src_monthly(key)

    out = d.copy()
    n = 0
    if kind in ("hourly", "weekly"):
        hcol = "hour_n" if "hour_n" in out.columns else "거래시간"
        for _, r in rules.iterrows():
            hrs = parse_range(r["범위"], kind)
            if kind == "hourly":
                col = f"예측_{r['계통']}"
            else:
                col = "예측"
            if col not in out.columns:
                continue
            if f"{col}_원본" not in out.columns:
                out[f"{col}_원본"] = out[col]
            m = out[hcol].isin(hrs)
            if kind == "weekly" and str(r["계통"]).startswith("D+"):
                m &= out["지평"].astype(str) == str(r["계통"])
            out.loc[m, col] = np.round(
                apply_value(out.loc[m, col].values, r["방식"], float(r["값"])), 2)
            n += int(m.sum())
    else:
        for _, r in rules.iterrows():
            col = parse_range(r["범위"], kind)
            if col not in out.columns:
                continue
            if f"{col}_원본" not in out.columns:
                out[f"{col}_원본"] = out[col]
            m = out["월"].astype(str) == str(key)
            out.loc[m, col] = np.round(
                apply_value(out.loc[m, col].values, r["방식"], float(r["값"])), 2)
            n += int(m.sum())

    #  보정량 열 — 원본 대비
    for c in [c for c in out.columns if c.endswith("_원본")]:
        base = c[:-3]
        if base in out.columns:
            out[f"{base}_보정량"] = np.round(out[base] - out[c], 2)

    stem, ext = os.path.splitext(src)
    dst = f"{stem}_보정{ext}"
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        for ln in head:
            f.write(ln + "\n")
        f.write("# ※ 본 파일은 협업 부서 수동 보정본입니다. "
                "모델 원본은 접미사 없는 파일을 참조하십시오.\n")
        out.to_csv(f, index=False)
    ap = applied_path(kind)
    os.makedirs(ADJ_DIR, exist_ok=True)
    rec = out.copy()
    rec.insert(0, "적용시각", f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    rec.insert(1, "대상키", str(key))
    rec.to_csv(ap, mode="a", header=not os.path.exists(ap),
               index=False, encoding="utf-8-sig")
    if not quiet:
        print(f"[적용] 규칙 {len(rules)}건 / 변경 {n}행")
        print(f"       보정본 → {os.path.relpath(dst, BASE_DIR)}")
        print(f"       원본 보존 → {os.path.relpath(src, BASE_DIR)}")
    return dst, len(rules), n


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="현업 수동 보정 엔진")
    ap.add_argument("--kind", default="hourly", choices=list(KINDS))
    ap.add_argument("--key", help="대상키 — 일자(YYYY-MM-DD)·기준일·월(YYYY-MM)")
    ap.add_argument("--target", default="smp_land", help="계통 또는 지평")
    ap.add_argument("--range", dest="rng", default="1-24",
                    help="시간대 구간(10-16)·개별(13,14)·월별 구분(평일·휴일·종합)")
    ap.add_argument("--mode", default="add", choices=list(MODES))
    ap.add_argument("--value", type=float, default=0.0)
    ap.add_argument("--reason", default="")
    ap.add_argument("--user", default="", help="담당자 성명 — 필수")
    ap.add_argument("--rule-id")
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    print("=" * 74)
    print(" step10_manual_adjust — 현업 수동 보정 v1.0")
    print(f" 실행시각 : {datetime.now():%Y-%m-%d %H:%M:%S} / 구분 : {a.kind}")
    print("=" * 74)

    if a.add:
        add_rule(a.kind, a.key, a.target, a.rng, a.mode, a.value,
                 a.reason, a.user)
    elif a.delete:
        delete_rule(a.kind, a.rule_id, a.user)
    elif a.apply:
        if not a.key:
            sys.exit("[중단] --key 필요")
        apply_rules(a.kind, a.key)
    else:
        list_rules(a.kind, a.key)
    print("=" * 74)


if __name__ == "__main__":
    main()
