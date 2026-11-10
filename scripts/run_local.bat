@echo off
REM ==============================================================
REM  SMP 예측 프로그램 로컬 운영 도구 (Windows)
REM  용도 : 더블클릭으로 예측·대시보드·점검·스케줄 메뉴 호출
REM  버전 : v1.0 (2026-08-16) / CP949 저장 - CMD 한글 출력 정합
REM ==============================================================
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_local.ps1" %*
endlocal
