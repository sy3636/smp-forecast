@echo off
REM ==============================================================
REM  SMP 익일 예측 자동 실행 배치
REM  용도 : Windows 작업 스케줄러 등록용(따옴표 중첩 회피)
REM  버전 : v1.2 (2026-08-12) / CP949 저장, 출력 인코딩 고정
REM  실행 : scripts\run_predict.bat
REM  로그 : logs\run_predict_YYYYMMDD.log
REM ==============================================================
setlocal

REM 프로젝트 루트(본 배치의 상위 폴더)로 이동
cd /d "%~dp0.."

REM 파이썬 출력 인코딩 고정 - 배치 리다이렉션 시 cp949 인코딩 오류 방지
REM  . 콘솔 미연결 상태에서 stdout이 cp949로 잡혀 특수문자 출력 실패
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM 일자별 로그 파일명 구성
for /f "tokens=1-3 delims=-" %%a in ("%date:~0,10%") do set TODAY=%%a%%b%%c
set TODAY=%TODAY:/=%
set LOGFILE=logs\run_predict_%TODAY%.log

if not exist logs mkdir logs

echo. >> "%LOGFILE%"
echo ============================================== >> "%LOGFILE%"
echo [%date% %time%] SMP 익일 예측 시작 >> "%LOGFILE%"

python scripts\step6_predict.py %* >> "%LOGFILE%" 2>&1
set RC=%ERRORLEVEL%

echo [%date% %time%] 종료코드 %RC% >> "%LOGFILE%"
endlocal & exit /b %RC%
