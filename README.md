# SMP Forecast

다음 날(D+1) **시간별 육지 SMP(전력시장가격)** 를 예측하고, **육지 급락 · 육지 급등 · 제주 급락** 경보를 내는 모델과 결과 대시보드입니다.

- 대시보드: **https://smp-forecast.onrender.com** (비밀번호 필요 — 팀 내부 공유)
- 기간: 학습 2020-01-01 ~ 2024-12-31 · 시험 2025-01-01 ~ 2026-10-31
- 시험 구간 가격 성적: MAE 7.02원 · rMAE 0.633 · 80% 예측 범위 포함률 80.3%

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `scripts/` | 데이터 수집(공공데이터 API) · 전처리 · 피처 · 학습 · 운영 예측 (`step1` ~ `step12`) |
| `experiments/ensemble/` | 가격 앙상블 · 경보 모델 · 사건 판정 · 대시보드 만들기(`build_my_dashboard.py`, `my_dashboard_template.html`) · 배포용 암호화(`encrypt_dashboard.py`) |
| `config/` | 기간 설정(`eval_settings.json`), API 주소, 모델 설정, 인증키 예시(`*.env.example`) |
| `outputs/ensemble/deploy/site/` | 배포용 페이지 — 대시보드를 비밀번호로 암호화한 `index.html` |
| `render.yaml` | Render 정적 사이트 설정 (`outputs/ensemble/deploy/site/` 만 배포) |
| `TEAM_REPRODUCE.md` | 처음부터 다시 만드는 순서 · 모델 레시피 · 검증 숫자 |

## 모델 요약

- **가격 (D+1 시간별, 전날 21시)**: Chronos-2 미세조정 + LightGBM 3종을 직전 180일 오차가 가장 작은 비율로 결합, 80% 예측 범위는 적응형 보정
- **육지 급락 경보 (전날 15시)**: 다음 날 11~15시 최저가를 모델 14개로 예측 → 중앙값 → 확률 보정, 휴일·주말에만 경보
- **육지 급등 경보 (전날 21시)**: 로지스틱 · LightGBM · RandomForest 앙상블 — 전일 가격·최근 급등 횟수·28일 대비 가격 차이 반영
- **제주 급락 경보 (전날 15시)**: KNN 회귀 · 제주 v1 · Chronos-2 확률 평균

자세한 설명은 [`TEAM_REPRODUCE.md`](TEAM_REPRODUCE.md) 5장과 `experiments/ensemble/RECIPE.md` 에 있습니다.

## 실행

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/ensemble/build_my_dashboard.py   # → outputs/ensemble/my_dashboard.html
bash scripts/publish_render.sh --no-build                       # 암호화 → GitHub 올리기 → Render 자동 배포
```

맥에서 LightGBM 이 `libomp` 를 못 찾으면 `DYLD_LIBRARY_PATH` 를 `.venv/lib/python3*/site-packages/sklearn/.dylibs` 로 지정합니다(`publish_render.sh` 는 자동으로 지정).

## 올리지 않은 것

공개 저장소이므로 다음은 올리지 않았습니다(`.gitignore`). 다시 만들려면 팀 내부에서 받은 데이터로 `TEAM_REPRODUCE.md` 순서를 따릅니다.

- `data/` — 원천·가공 데이터 (공공데이터포털 API 로 다시 받을 수 있음)
- `outputs/` 의 모델 결과(`.pkl` · `.csv`)와 원본 대시보드(`my_dashboard.html`)
- 인증키(`config/.env`) — 각자 발급, 예시는 `config/.env.example`

---
한국중부발전(주) 동반성장부 SMP 예측 프로그램 기반 · 교육·연구 목적
데이터 출처: 한국전력거래소 · 기상청 · 한국천문연구원 (공공데이터포털)
