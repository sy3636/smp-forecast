@echo off
REM ==============================================================
REM  SMP 예측 통합 대시보드 실행 배치
REM  용도 : Streamlit 대시보드 기동 (가상환경 자동 인식)
REM  버전 : v1.1 (2026-08-12) / CP949 저장 - CMD 한글 출력 정합
REM  실행 : scripts\run_dashboard.bat
REM  접속 : http://localhost:8501
REM  종료 : 콘솔 창에서 Ctrl+C
REM ==============================================================
setlocal

cd /d "%~dp0.."

REM 파이썬 출력 인코딩 고정 - 로그 리다이렉션 시 오류 방지
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import streamlit, plotly" 2>nul
if errorlevel 1 (
    echo [중단] streamlit 또는 plotly 미설치
    echo        설치 : "%PY%" -m pip install -r requirements.txt
    exit /b 1
)

echo ==============================================================
echo  SMP 예측 통합 대시보드
echo  접속 주소 : http://localhost:8501
echo  종료 방법 : 본 창에서 Ctrl+C
echo ==============================================================

"%PY%" -m streamlit run src\app\dashboard.py --server.port 8501 %*

endlocal
