#!/usr/bin/env bash
# SMP Forecast — 대시보드 다시 만들기 → 비밀번호로 암호화 → GitHub 올리기 (Render 가 자동으로 다시 배포)
#
# 사용 (프로젝트 폴더에서):
#   bash scripts/publish_render.sh              대시보드를 다시 만든 뒤 암호화·올리기
#   bash scripts/publish_render.sh --no-build   이미 만든 대시보드를 암호화·올리기만
#
# 저장소 = 프로젝트 폴더 전체 (.gitignore 로 데이터·결과·인증키는 제외, 코드 + 암호화 페이지만 올라감)
# 비밀번호(알파벳 12자리)는 두 번 묻고, 어디에도 저장하지 않음 — 매번 같은 비밀번호를 넣을 것
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SITE=outputs/ensemble/deploy/site
[ -x "$PY" ] || { echo "[중단] .venv 가 없습니다. README.md 「실행」대로 만들어 주세요."; exit 1; }
git remote get-url origin >/dev/null 2>&1 || { echo "[중단] GitHub 저장소 주소(origin)가 연결되지 않았습니다."; exit 1; }

# 맥: LightGBM 이 쓰는 libomp 를 scikit-learn 에 들어 있는 것으로 대신 불러옴
DYLIBS="$(ls -d .venv/lib/python3*/site-packages/sklearn/.dylibs 2>/dev/null | head -1 || true)"
[ -n "$DYLIBS" ] && export DYLD_LIBRARY_PATH="$DYLIBS"

if [ "${1:-}" != "--no-build" ]; then
  echo "== 1/3 대시보드 만들기"
  "$PY" experiments/ensemble/build_my_dashboard.py
fi

echo "== 2/3 비밀번호로 암호화"
"$PY" experiments/ensemble/encrypt_dashboard.py --out "$SITE/index.html"

echo "== 3/3 GitHub 에 올리기 (코드 바뀐 것도 함께)"
git add -A
if git diff --cached --quiet; then
  echo "바뀐 내용이 없습니다."
  exit 0
fi
git commit -q -m "대시보드 갱신 $(date '+%Y-%m-%d %H:%M')"
git push -u origin main
echo "완료 — Render 가 1~2분 안에 자동으로 다시 배포합니다."
