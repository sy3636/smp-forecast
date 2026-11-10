# -*- coding: utf-8 -*-
"""
SMP 예측 프로그램 - 산출물 등가성 대조
================================================
목적 : 두 실행 환경(로컬 Windows / 컨테이너 Linux)의 주간 예측 산출물을
       행 단위로 대조하여 완전 일치 여부를 판정

배경 : PowerShell 파이프로 코드를 전달하면 한글 컬럼명이 손실되므로
       (2026.8.18 실측 — '지평' → '??') 대조 로직을 별도 파일로 분리함

사용 : python scripts\\compare_outputs.py {A파일} {B파일} [A명칭] [B명칭]

종료코드 : 0 = 완전 일치 / 1 = 불일치 또는 오류
버전 : v1.0 (2026-08-18)
"""

import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 미설치")

KEY = ["지평", "대상일", "거래시간"]
VAL = "예측"


def main():
    if len(sys.argv) < 3:
        sys.exit("사용법: compare_outputs.py {A파일} {B파일} [A명칭] [B명칭]")
    pa, pb = sys.argv[1], sys.argv[2]
    na = sys.argv[3] if len(sys.argv) > 3 else "A"
    nb = sys.argv[4] if len(sys.argv) > 4 else "B"

    try:
        a = pd.read_csv(pa, encoding="utf-8-sig")
        b = pd.read_csv(pb, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [중단] 파일 읽기 실패 — {e}")
        return 1

    miss = [c for c in KEY + [VAL] if c not in a.columns or c not in b.columns]
    if miss:
        print(f"  [중단] 필수 컬럼 누락 : {miss}")
        return 1

    print(f"  행수        : {na} {len(a)} / {nb} {len(b)}")

    m = a[KEY + [VAL]].merge(b[KEY + [VAL]], on=KEY, suffixes=("_a", "_b"))
    if m.empty:
        print("  [중단] 대조 가능한 공통 행 없음")
        return 1

    d = (m[f"{VAL}_a"] - m[f"{VAL}_b"]).abs()
    same = int((d < 1e-9).sum())
    print(f"  대조 행수   : {len(m)}")
    print(f"  최대 절대차 : {d.max():.6f} 원/kWh")
    print(f"  평균 절대차 : {d.mean():.6f} 원/kWh")
    print(f"  완전 일치   : {same} / {len(m)} 행")
    print()
    print("  지평별 평균 대조")
    g = m.groupby("지평").agg(A=(f"{VAL}_a", "mean"), B=(f"{VAL}_b", "mean"))
    for idx, r in g.iterrows():
        gap = abs(r["A"] - r["B"])
        print(f"    {idx:<5} {na} {r['A']:>7.2f} / {nb} {r['B']:>7.2f} / 차이 {gap:.4f}")

    if d.max() >= 1e-9:
        print()
        print("  차이 상위 5행")
        t = m.assign(차이=d).nlargest(5, "차이")
        for _, r in t.iterrows():
            print(f"    {r['지평']} {r['대상일']} {int(r['거래시간']):>2}시 : "
                  f"{r[f'{VAL}_a']:.2f} vs {r[f'{VAL}_b']:.2f} (차 {r['차이']:.4f})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
