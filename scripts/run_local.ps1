# ==============================================================================
# SMP 예측 프로그램 - 로컬(Windows) 운영 도구
# ==============================================================================
# 목적 : 예측 실행·대시보드·점검·스케줄 관리를 메뉴로 처리하여 명령 암기 부담 제거
#
# 실행 : scripts\run_local.bat 더블클릭
#        또는 powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
#        인자 지정 시 메뉴 없이 즉시 실행 : ... run_local.ps1 weekly
#
# 버전 : v1.0 (2026-08-16)
# ==============================================================================

param([string]$Mode = "")

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# 가상환경 자동 인식
$Py = "python"
$venv = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $venv) { $Py = $venv }

$Predict = Join-Path $Root "outputs\predict"
$Forecast = Join-Path $Root "outputs\forecast"

function Line { Write-Host ("=" * 66) }

function Run-Py([string]$script, [string[]]$a = @()) {
    Write-Host "  > $script $($a -join ' ')" -ForegroundColor DarkGray
    & $Py (Join-Path $Root "scripts\$script") @a
    if ($LASTEXITCODE -eq 0) { Write-Host "`n  [O] 정상 종료" -ForegroundColor Green }
    else {
        Write-Host "`n  [X] 종료코드 $LASTEXITCODE — 위 오류 메시지를 확인하십시오" -ForegroundColor Red
        Write-Host "      자주 발생하는 원인 : 데이터 미갱신 / 모델 파일 부재 / 라이브러리 버전" -ForegroundColor DarkGray
    }
}

# 산출물 최신일 조회 — 파일명에서 일자 추출
function Latest([string]$prefix, [string]$dir) {
    $f = Get-ChildItem $dir -Filter "$prefix*.csv" -ErrorAction SilentlyContinue
    if (-not $f) { return $null }
    ($f | ForEach-Object { if ($_.Name -match '(\d{6,8})') { $matches[1] } } | Sort-Object | Select-Object -Last 1)
}

function Show-Status {
    $today = Get-Date -Format "yyyyMMdd"
    $tomo = (Get-Date).AddDays(1).ToString("yyyyMMdd")
    $thism = Get-Date -Format "yyyyMM"

    $w = Latest "SMP주간예측_" $Predict
    $h = Latest "SMP예측_" $Predict
    $m = Latest "SMP월별전망_" $Forecast

    function Mark($v, $expect) {
        if (-not $v) { return "[X] 없음" }
        if ($v -ge $expect) { return "[O] $v" } else { return "[!] $v" }
    }

    Write-Host ""
    Write-Host "  산출물 현황" -ForegroundColor Cyan
    Write-Host ("  {0,-16}{1,-16}{2}" -f "구분", "기대", "실제")
    Write-Host "  " + ("-" * 48)
    Write-Host ("  {0,-16}{1,-16}{2}" -f "주간 예측", $today,  (Mark $w $today))
    Write-Host ("  {0,-16}{1,-16}{2}" -f "시간별 예측", $tomo, (Mark $h $tomo))
    Write-Host ("  {0,-16}{1,-16}{2}" -f "월별 전망", $thism,  (Mark $m $thism))
    Write-Host ""
    Write-Host "  [O] 최신 / [!] 갱신 필요 / [X] 산출물 없음" -ForegroundColor DarkGray
}

function Show-Menu {
    Line
    Write-Host " SMP 예측 프로그램 로컬 운영 (Windows)"
    Line
    Show-Status
    Line
    Write-Host "  [1] 주간 예측        D+1~D+6 산출 (주력)"
    Write-Host "  [2] 시간별 예측      D+1 24시간·사후검증  ※ 23시 이후 권장"
    Write-Host "  [3] 월별 전망        향후 12개월"
    Write-Host ""
    Write-Host "  [4] 대시보드 실행    http://localhost:8501"
    Write-Host "  [5] 최근 로그 보기   실행 결과 확인"
    Write-Host ""
    Write-Host "  [6] 모델 재학습      지평별 6종 (수 분 소요)"
    Write-Host "  [7] 제주 동일성 감시  월 1회 점검"
    Write-Host "  [8] 산출물 백업      무결성 검증 포함"
    Write-Host ""
    Write-Host "  [9] 스케줄 등록·조회  자동 실행 설정"
    Write-Host "  [0] 종료"
    Line
}

function Do-Action([string]$m) {
    switch ($m) {
        "weekly"  { Write-Host "`n■ 주간 예측 (D+1~D+6)`n"; Run-Py "step6_weekly_predict.py" }
        "daily"   {
            Write-Host "`n■ 시간별 예측`n"
            if ((Get-Date).Hour -lt 23) {
                Write-Host "  [주의] 현재 $((Get-Date).Hour)시 — D+1 확정값은 23시경 공개됩니다" -ForegroundColor Yellow
                Write-Host "         지금 실행하면 수요예측 미확보로 실패할 수 있습니다"
                if ((Read-Host "  계속하시겠습니까? (y/N)") -ne "y") { Write-Host "  취소"; return }
            }
            Run-Py "step6_predict.py"
        }
        "monthly" { Write-Host "`n■ 월별 전망`n";           Run-Py "step7_monthly_forecast.py" }
        "dash"    {
            Write-Host "`n■ 대시보드 실행 — 종료는 Ctrl+C`n"
            if (-not $env:SMP_DASH_PW) { Write-Host "  (암호 미설정 — 인증 없이 접속)" -ForegroundColor DarkGray }
            Write-Host "  접속 : http://localhost:8501`n" -ForegroundColor Green
            & $Py -m streamlit run (Join-Path $Root "src\app\dashboard.py") --server.port 8501
        }
        "log"     {
            Write-Host "`n■ 최근 로그`n"
            $f = Get-ChildItem (Join-Path $Root "logs") -Filter "run_*.log" |
                 Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($f) {
                Write-Host "  파일 : $($f.Name)  ($($f.LastWriteTime))`n" -ForegroundColor DarkGray
                Get-Content $f.FullName -Tail 30
            } else { Write-Host "  로그 없음" -ForegroundColor Yellow }
        }
        "train"   { Write-Host "`n■ 모델 재학습 — 수 분 소요`n"; Run-Py "step4_train_horizon.py" @("--target","smp_land") }
        "jeju"    { Write-Host "`n■ 제주 동일성 감시`n";        Run-Py "step2_check_jeju_parity.py" }
        "backup"  {
            # 하위명령 방식(backup/restore/verify) — 플래그 아님(2026.8.21 정정)
            # 원자료 포함 : data/raw/holiday 누락 시 days_to_holiday 결함 재발
            Write-Host "`n■ 산출물 백업 — 원자료 포함`n"
            Run-Py "migrate_state.py" @("backup","--with-raw")
        }
        "sched"   {
            Write-Host "`n■ 자동 실행 스케줄`n"
            Write-Host "  --- 현재 등록 상태 ---"
            $q = cmd /c "schtasks /query /fo list 2>&1" | Select-String "SMP"
            if ($q) { $q | ForEach-Object { Write-Host "   $_" } }
            else { Write-Host "   등록된 SMP 작업 없음" -ForegroundColor Yellow }

            Write-Host "`n  [1] 등록(또는 재등록)   [2] 해제   [Enter] 돌아가기"
            switch (Read-Host "  선택") {
                "1" {
                    cmd /c "schtasks /create /tn `"SMP주간예측`" /tr `"$Root\scripts\run_weekly.bat`" /sc daily /st 15:00 /f"
                    cmd /c "schtasks /create /tn `"SMP익일예측`" /tr `"$Root\scripts\run_predict.bat`" /sc daily /st 23:30 /f"
                    Write-Host "`n  등록 완료 — taskschd.msc 에서 다음 2건을 추가 설정하십시오" -ForegroundColor Green
                    Write-Host "   · 조건 탭 : 'AC 전원인 경우에만 시작' 체크 해제"
                    Write-Host "   · 설정 탭 : '놓친 경우 가능한 한 빨리 시작' 체크"
                }
                "2" {
                    cmd /c "schtasks /delete /tn `"SMP주간예측`" /f"
                    cmd /c "schtasks /delete /tn `"SMP익일예측`" /f"
                    Write-Host "`n  해제 완료" -ForegroundColor Green
                }
            }
        }
        default   { Write-Host " 알 수 없는 모드 : $m" -ForegroundColor Red }
    }
}

$map = @{ "1"="weekly"; "2"="daily"; "3"="monthly"; "4"="dash"; "5"="log";
          "6"="train"; "7"="jeju"; "8"="backup"; "9"="sched" }

# 인자 지정 시 즉시 실행 — 번호(1~9)와 모드명 모두 허용
if ($Mode) {
    if ($map.ContainsKey($Mode)) { $Mode = $map[$Mode] }
    Do-Action $Mode
    exit 0
}

while ($true) {
    Show-Menu
    $sel = Read-Host " 선택"
    if ($sel -eq "0") { Write-Host " 종료합니다."; break }
    if ($map.ContainsKey($sel)) {
        Do-Action $map[$sel]
        Write-Host ""
        Read-Host " 엔터를 누르면 메뉴로 돌아갑니다" | Out-Null
    } else {
        Write-Host " 0~9 중에서 선택하십시오" -ForegroundColor Yellow
        Start-Sleep -Seconds 1
    }
}
