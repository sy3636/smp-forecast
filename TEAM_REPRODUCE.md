# SMP 예측 대시보드 — 팀원 재현 안내

이 문서를 따라 하면 이 대시보드(`outputs/ensemble/my_dashboard.html`)를 각자 컴퓨터에서 똑같이 만들 수 있다.
기준일 2026-11-11 (대시보드 화면 개편 반영) · 「한국중부발전(주) 동반성장부 SMP 예측 프로그램 기반」

---

## 빠른 시작 (압축 파일 `smp_student_team.zip`을 받았을 때)

```bash
cd student-practice-team
python3 -m venv .venv
.venv/bin/pip install pandas==3.0.5 numpy==2.5.2 lightgbm==4.7.0 scikit-learn==1.9.0 catboost==1.2.10 xgboost==3.4.1 scipy pyarrow requests
.venv/bin/python experiments/ensemble/build_my_dashboard.py      # → outputs/ensemble/my_dashboard.html
```

1. 압축을 푼다.
2. 위 명령을 실행한다.
3. `outputs/ensemble/my_dashboard.html`을 Chrome·Safari 같은 브라우저로 연다(7장).

결과 파일이 압축에 다 들어 있어서 1~2분이면 끝난다. 모델 결과까지 처음부터 다시 만들려면 4장을 따른다.

## 0. 먼저 알아 둘 것

- **모델을 다시 학습하지 않는다.** 가격 모델(LightGBM·Chronos-2)은 팀 학습 결과 파일(`.pkl`)을 그대로 쓴다.
  `run_chronos.py`, `run_chronos_ft.py`, `run_lgbm.py`는 실행하지 않는다. 실행하면 결과가 달라진다.
  경보 모델은 스크립트 안에서 매달 다시 학습하지만, 그 이전 자료만 쓰므로 실행할 때마다 같은 결과가 나온다.
- **외부 공개 금지.** 데이터, 결과 CSV·HTML, `.pkl`은 팀 안에서만 주고받는다. 공개 저장소에 올리지 않는다.
- **인증키는 각자 발급한다.** `config/.env`는 주고받지 않는다. 인증키를 화면·문서·커밋에 남기지 않는다.
- **시험 구간 결과를 보고 모델이나 기준을 바꾸지 않는다.** 모든 선택은 학습 구간 성적으로만 한다.

## 1. 기간 설정

모든 기간은 `config/eval_settings.json` 한 곳에서 읽는다.

| 항목 | 값 |
|---|---|
| 전체 데이터 | 2020-01-01 ~ 2026-10-31 (모든 결과가 10-31에서 끝남) |
| 판정 시작 | 2020-01-29 (직전 28일 기준값이 필요해서) |
| 학습 | 2020-01-01 ~ 2024-12-31 (가격 예측은 2022년부터 있어서 실제 채점은 2022–2024) |
| 시험 | 2025-01-01 ~ 2026-10-31 (669일) |
| 예측 시점 | 가격·육지 급등: 전날 21시 · 육지 급락·제주 급락: 전날 15시 |

## 2. 준비물

### 2-1. 폴더
이 작업 폴더(`student-practice-team`)를 팀 내부 경로로 통째로 받는다. 단, 아래 두 가지는 빼고 받는다.
- `.venv/` — 컴퓨터마다 새로 만든다.
- `config/.env` — 각자 만든다.

꼭 있어야 하는 파일은 아래와 같다.

| 종류 | 파일 |
|---|---|
| 데이터 | `data/processed/smp_dataset_hourly.csv`(2020-01-01부터), `smp_features_smp_land.csv` · `data/interim/`의 `api_smp_demand_hourly.csv`, `api_smp_recent.csv`, `api_smp_fill.csv`, `asos_temp_recent.csv`, `api_holiday.csv`, `merit_features_monthly.csv`, `src_merit_public.csv`, `fuel_index_monthly.csv`, `demand_own_forecast.csv` · `data/external/fuel_index_fred_2020_20261108.csv` |
| 가격 모델 결과 (팀 학습본) | `outputs/ensemble/chronos_off.pkl`, `chronos_dem_off.pkl`, `chronos_ft_lora.pkl`, `pred_lgb_l1_weekly.pkl`, `pred_lgb_stack_weekly.pkl`, `pred_lgb_stack_ft_weekly.pkl` |
| 설정 | `config/eval_settings.json` |
| 코드 | `experiments/ensemble/*.py`, `my_dashboard_template.html`, `scripts/step3_extend_2020.py` |

> `api_smp_fill.csv`는 교육용 데이터에 빠져 있던 2026-07-31 실적을 SMP API로 받아 채운 파일이다. 이 파일이 없으면 2026년 급등 구간 수가 2개 적게 나온다.

### 2-2. 파이썬 환경
Python 3.12 이상(작성자는 3.14)에서 아래를 설치한다.

```bash
cd student-practice-team
python3 -m venv .venv
.venv/bin/pip install pandas==3.0.5 numpy==2.5.2 lightgbm==4.7.0 scikit-learn==1.9.0 \
    catboost==1.2.10 xgboost==3.4.1 scipy pyarrow requests cryptography
```

- Windows는 `.venv/bin/python` 대신 `.venv\Scripts\python`을 쓴다.
- `torch`와 `chronos`는 필요 없다. Chronos-2는 결과 파일(`.pkl`)만 쓴다.
- 패키지 버전이 다르면 소수점 끝자리가 조금 달라질 수 있다.
- 맥에서 `Library not loaded: @rpath/libomp.dylib`(LightGBM) 오류가 나면, scikit-learn에 들어 있는 libomp를 쓰도록 지정하고 실행한다.
  `export DYLD_LIBRARY_PATH=$(ls -d .venv/lib/python3*/site-packages/sklearn/.dylibs)`

### 2-3. 인증키 (데이터를 새로 받을 때만)
폴더에 데이터가 이미 있으면 필요 없다. 2020년 자료를 직접 다시 받을 때만 아래를 한다.
1. [공공데이터포털](https://www.data.go.kr)에서 다음 API를 활용신청한다.
   - 한국전력거래소_계통한계가격 및 수요예측
   - 기상청_ASOS 시간자료
   - 한국천문연구원_특일 정보
2. `config/.env` 파일을 만들고 `KPX_API_KEY=발급받은키` 한 줄을 넣는다. 수집 스크립트는 이름이 `_API_KEY`로 끝나는 항목을 읽는다.
   - 개발계정은 하루 100건까지만 호출할 수 있다.

## 3. 데이터 만들기 (이미 있으면 건너뜀)

2021년 이후 데이터는 그대로 두고, 2020년 행만 앞에 붙인다.

```bash
.venv/bin/python scripts/step1_collect_weather.py --start 20200101 --end 20201231            # ASOS 96건
.venv/bin/python scripts/step1_collect.py --source holiday --start 20200101 --end 20201231    # 특일 12건
.venv/bin/python scripts/step3_extend_2020.py --check   # 2021년을 같은 규칙으로 만들어 기존 데이터와 같은지 검사
.venv/bin/python scripts/step3_extend_2020.py           # 2020년 행 추가
```

| 2020년 자료 | 출처 |
|---|---|
| SMP·하루전 수요예측 | 이미 있는 `api_smp_demand_hourly.csv` (전력거래소 API) |
| 기온 | 기상청 ASOS API (8지점 × 12달) |
| 공휴일 | 한국천문연구원 특일 정보 API |
| 국제 연료지수 | FRED (유가·LNG·석탄·환율) |
| 전력수요 실적 · 구름 · 변동비순위표 | 2020년 자료 없음 — 모델에서 안 쓰거나 원래 2022-03부터 있음 |

## 4. 결과 만들기 (순서대로, 모두 `experiments/ensemble/`에서)

```bash
cd experiments/ensemble
PY=../../.venv/bin/python
$PY ensemble.py --final        # 가격 앙상블 + 80% 범위 → price_final.pkl
$PY warnings_model.py --final  # 팀 경보 모델(제주 v1 변수용) → warnings_final.pkl
$PY export_results.py          # 결과 CSV
$PY explain.py                 # 예측 근거 문장 (~6분)
$PY build_overview.py
$PY warnings_weekday.py        # 평일 급락(하루 화면용)
$PY warnings_jeju.py           # 제주 급락 v1 (v2의 성분)
$PY events.py                  # 사건 구간 판정 → events_*.csv
$PY warn_search.py             # 경보 후보 확률 → warn_search_probs.pkl (제주 v2가 씀, ~5–10분)
$PY warnings_spike.py          # 육지 급등 경보
$PY warnings_crash.py          # 육지 급락 경보 v2 (~5분)
$PY warnings_jeju_v2.py        # 제주 급락 경보 v2 (~5분)
$PY events.py                  # 새 경보로 다시 채점
$PY build_my_dashboard.py      # → outputs/ensemble/my_dashboard.html
```

화면만 고쳤을 때는 마지막 `build_my_dashboard.py`만 다시 실행하면 된다.

## 5. 모델 레시피

### 5-1. 가격 예측 (D+1 시간별 육지 SMP, 전날 21시)
- **성분 4개**
  - Chronos-2 미세조정 (LoRA, 매년 재학습)
  - LightGBM 보정형 2개: 미세조정 Chronos 분위수를 입력으로 받는 것 하나, 제로샷 Chronos 분위수를 입력으로 받는 것 하나
  - LightGBM 기본형
  - LightGBM 공통: 변수 72개, L1 손실, asinh 변환, 매주 재학습
- **결합**: 매달 1일, 직전 180일 오차합이 가장 작은 비율로 섞는다(합 1, 음수 없음).
- **80% 범위**: 결합 예측에 직전 91일 잔차 분위수 × Chronos 분위수 폭을 더하고 뺀다. 적응형 보정(ACI)을 쓴다.
- 자세한 설명은 `experiments/ensemble/RECIPE.md`에 있다.

### 5-2. 육지 급락 경보 (전날 15시) — `warnings_crash.py`
- **방식**: 다음 날 11~15시 최저가를 모델 14개로 예측한다.
  1. 14개 예측의 **중앙값**을 낸다.
  2. 매달 로지스틱으로 보정해 확률로 바꾼다.
  3. 휴일·주말에만 「경보」 한 단계를 낸다(확률 ≥ 0.733).
- **모델 14개**: LightGBM(중앙값·Huber) · CatBoost(중앙값·RMSE) · XGBoost · HistGB · RandomForest · ExtraTrees · Ridge · Huber · KNN · SVR · MLP · Chronos-2(15시)
- **재학습**: 매달 1일, 그 이전 자료로만 한다.
- **고른 근거**: 후보 29개 중 학습 구간 헛경보가 가장 적었다.

### 5-3. 육지 급등 경보 (전날 21시) — `warnings_spike.py`
- **라벨**: 그날 급등 구간(같은 시각 28일 중앙값 + 20원)이 있으면 1이다.
- **방식**: 로지스틱(C=0.05) 50% · LightGBM 분류(잎 7) 30% · RandomForest 20%
- **단계**: 학습 구간 사례의 95%를 잡는 확률이 「주의」, 80%를 잡는 확률이 「경보」다(고정 기준). 매달 1일 재학습한다.

### 5-4. 제주 급락 경보 (전날 15시) — `warnings_jeju_v2.py`
- **방식**: 확률 3개를 평균한다.
  - KNN 회귀(제주 하루 최저가 예측 → 확률 보정)
  - 제주 v1 모델(로지스틱 10% · 최근 빈도 20% · 육지 급락 신호 70%)
  - Chronos-2 제로샷 하루 최저가 예측
- **단계**: 「경보」 한 단계다. 학습 구간 고정 기준으로 휴일 ≥ 0.689, 평일 ≥ 0.526이다.

## 6. 잘 만들어졌는지 확인

`build_my_dashboard.py`가 출력하는 숫자와 화면의 숫자가 아래와 같으면 성공이다.

**사건 수 검증** (`events.py` 실행 시 출력, 2020-01-29 ~ 2026-10-31, 휴일 / 평일 구간 수)
- 육지 급등 284 / 1,199 (합 1,483) · 육지 급락 46 / 3 · 제주 0원 이하 62 / 72 · 급등 있는 날 833일

**가격 (시험 구간)**
- MAE 7.02원 · rMAE 0.6330 · MAPE 7.35% · 편의 +1.73% · 80% 범위 포함률 80.3%

**사건 판정 결과 (시험 구간, 확률(경보) 모델)**

| 사건 | 구분 | 구간 수 | 성공 | 부분 | 실패 | 헛경보(날) |
|---|---|---|---|---|---|---|
| 육지 급등 | 휴일 | 109 | 64 | 33 | 12 | 74 (주의 56 · 경보 18) |
| | 평일 | 429 | 353 | 72 | 4 | 197 (주의 102 · 경보 95) |
| 육지 급락 | 휴일 | 36 | 29 | 0 | 7 | 60 |
| | 평일 | 3 | 0 | 0 | 3 | 0 |
| 제주 급락 | 휴일 | 52 | 39 | 0 | 13 | 54 |
| | 평일 | 56 | 44 | 0 | 12 | 150 |

숫자가 다르면 아래 순서로 확인한다.
1. `api_smp_fill.csv`가 있는지
2. `eval_settings.json`의 기간이 맞는지
3. 4장의 실행 순서를 지켰는지

## 7. 화면 보기

`outputs/ensemble/my_dashboard.html`을 **Chrome·Safari 같은 브라우저**로 바로 연다(파일을 두 번 클릭).
- VS Code의 HTML 미리보기처럼 자바스크립트를 막는 창에서는 화면 틀만 보인다(화면에 안내 문구가 뜬다).
- 템플릿(`experiments/ensemble/my_dashboard_template.html`)을 열면 결과 데이터가 없어 화면 틀만 보인다. 반드시 `build_my_dashboard.py`가 만든 `outputs/ensemble/my_dashboard.html`을 연다.
- VS Code 안에서 보려면 `.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory outputs/ensemble`을 실행하고 `Simple Browser: Show`로 `http://127.0.0.1:8765/my_dashboard.html`을 연다.

**화면 구성** — 제목 「SMP Forecast」 오른쪽의 메뉴 3개로 화면을 나눠 본다. 주소 끝(`#overview` · `#day` · `#score`)에 지금 화면이 남아 새로고침·뒤로 가기를 해도 유지된다.

1. **한눈에 보기**
   - 경보 칸 3개: 육지 급락 · 육지 급등 · 제주 급락 — 시험 구간 적중 구간 수(파랑 = 급락, 빨강 = 급등), 헛경보 날 수, 놓친 구간 수
   - 가격 칸 3개: 가격 예측 평균 오차(MAE) · 80% 예측 범위 포함률 · 예측근거(최근 4주 영향 1~3위 막대)
   - 일자별 평균 가격: 날짜를 누르면 「하루 자세히」로 넘어간다.
   - 월별 달력: 오른쪽 버튼으로 연도(2025년 · 2026년)를 고른다. 칸 색이 진할수록 그날 평균 오차가 크고, 점은 실제 급락(파랑) · 급등(빨강)이다. 날짜를 누르면 「하루 자세히」로 넘어간다.
2. **하루 자세히** — 날짜 고르기(◀ 날짜 ▶ · 실제 사건일로 이동)와 오른쪽 「육지 | 제주」 버튼
   - 육지: 가격 · 수요예측 · 태양광 숫자, 24시간 실적 · 예측 · 80% 범위 · 급등 기준선 그래프, 육지 급락 · 급등 경보, 예측 근거(문장 + 요인별 기여 막대)
   - 제주: 제주 가격 · 0원 이하 시간 · 육지와 비교 숫자, 제주 24시간 실적 · 예측 그래프(급락 기준 0원 선), 제주 급락 경보
     - 제주 예측은 운영 예측 이력(`outputs/predict/predictions_smp_jeju.csv`)이 있으면 그것을, 없으면 기본 mirror 방식(육지 최종 예측을 제주에 복사)을 쓴다. mirror 값은 독립 제주 모델 예측이 아니다.
   - 그래프에 마우스를 올리면 그 시각의 실적 · 예측 등이 뜬다.
   - 날짜 옆 「표로 보기」를 누르면 그날 시간별 표가 팝업으로 뜬다(지금 고른 지역).
3. **성적과 기준**
   - 사건 판정 결과: 「사건 판정」 표(성공(경보) · 부분(주의) · 실패(놓침)), 「헛경보」 표, 「실제 발생일과 예측일 비교」(사건 · 휴일/평일별 표 6개, 예측과 실제가 맞은 대각선 칸 = 적중, 분홍)
     - 모델(확률 · 가격)과 시험 기간을 바꾸면 세 가지 모두 다시 계산한다.
     - 「실제 발생일과 예측일 비교」는 날짜 단위다. 확률 모델은 주의 이상을, 가격 모델은 하루 중 한 번이라도 사건 문턱을 넘은 예측을 「예측 있음」으로 센다. 제주 가격 모델은 제주 시간별 예측 이력 또는 mirror 대체 예측을 쓰며, 사건 판정표의 성공 · 부분 · 실패는 기존 경보 모델 채점을 따른다.
     - CSV 내려받기와 표 복사에 세 가지가 모두 들어간다.
   - 가격 목표 판정 · 월별 성적 · 판정 기준과 읽는 법

## 8. 알려진 한계
- 가격 모델(`.pkl`)은 2021년부터 학습된 팀 결과다.
  - 2020년부터 다시 학습해 비교해 봤다. 학습 구간은 좋아졌지만 시험 구간 성적이 거의 같아서 바꾸지 않았다.
  - 비교 결과는 `outputs/ensemble/price_compare_report.json`에 있다.
- 2026-11 이후로 늘리려면 팀에서 새 학습 결과 파일을 받아야 한다.
- 학습 구간의 급락·제주 0원 사례가 적다(휴일 급락 10일, 제주 0원 14일). 그래서 경보 기준값이 불안정할 수 있다.
