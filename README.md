# SMP Forecast — 배포용 저장소

팀 내부 공유용 SMP 예측 대시보드를 Render 무료 정적 사이트(Static Site)로 배포하는 저장소입니다.

- 이 저장소에는 **비밀번호로 암호화된 페이지 한 장**(`site/index.html`)만 들어 있습니다. 비밀번호를 모르면 소스 보기로도 내용을 볼 수 없습니다(PBKDF2-SHA256 60만 회 → AES-256-GCM).
- 원본 대시보드, 데이터, 모델 결과(`.csv`·`.pkl`)는 올리지 않습니다. 저장소는 **비공개(Private)** 로 둡니다.
- 비밀번호는 이 저장소 어디에도 저장하지 않습니다. 팀원에게는 따로 전달합니다.

## 폴더

| 파일 | 역할 |
|---|---|
| `site/index.html` | 암호화된 대시보드 (로그인 화면 → 비밀번호가 맞으면 브라우저 안에서 풀림) |
| `site/robots.txt` | 검색 엔진 수집 막기 |
| `render.yaml` | Render 설정 (정적 사이트, `site/` 만 배포, 검색 제외·캐시 헤더) |

## 대시보드 갱신하기

프로젝트 폴더(`student-practice-team`)에서:

```bash
bash scripts/publish_render.sh              # 대시보드 다시 만들기 → 암호화 → GitHub 올리기
bash scripts/publish_render.sh --no-build   # 이미 만든 대시보드를 암호화해서 올리기만
```

비밀번호(알파벳 12자리)를 두 번 묻습니다. **매번 같은 비밀번호**를 넣어야 팀원이 쓰던 비밀번호가 그대로 통합니다.
GitHub 에 올라가면 Render 가 1~2분 안에 자동으로 다시 배포합니다.
