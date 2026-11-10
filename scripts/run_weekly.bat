@echo off
REM ==============================================================
REM  SMP 주간 예측 자동 실행 배치 (D+1 ~ D+6)
REM  용도 : Windows 작업 스케줄러 등록용 - 매일 14:00 권장
REM  근거 : SMP 확정값은 15~23시 사이 공개되므로 그 이전 실행 시
REM         D+1도 실질 예측 가치를 가짐 (2026.8.12 실측)
REM  버전 : v1.0 (2026-08-12) / CP949 저장
REM  실행 : scripts\run_weekly.bat
REM  로그 : logs\run_weekly_YYYYMMDD.log
REM ==============================================================
setlocal

cd /d "%~dp0.."

REM 파이썬 출력 인코딩 고정 - 배치 리다이렉션 시 오류 방지
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

for /f "tokens=1-3 delims=-" %%a in ("%date:~0,10%") do set TODAY=%%a%%b%%c
set TODAY=%TODAY:/=%
set LOGFILE=logs\run_weekly_%TODAY%.log

if not exist logs mkdir logs

echo. >> "%LOGFILE%"
echo ============================================== >> "%LOGFILE%"
echo [%date% %time%] SMP 주간 예측 시작 >> "%LOGFILE%"

python scripts\step6_weekly_predict.py %* >> "%LOGFILE%" 2>&1
set RC=%ERRORLEVEL%

echo [%date% %time%] 종료코드 %RC% >> "%LOGFILE%"
endlocal & exit /b %RC%
