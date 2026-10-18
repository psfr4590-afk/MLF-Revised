$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$fail = 0
Write-Host "Model Lab PREFLIGHT" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

function Check($label, $ok, $detail="") {
  if ($ok) { Write-Host "[PASS] $label $detail" -ForegroundColor Green }
  else { Write-Host "[FAIL] $label $detail" -ForegroundColor Red; $script:fail++ }
}

Check "Python launcher" ([bool](Get-Command python -ErrorAction SilentlyContinue))
Check "run_pipeline.py" (Test-Path (Join-Path $Root "run_pipeline.py"))
Check "run_command_center.py" (Test-Path (Join-Path $Root "run_command_center.py"))
Check "launch.py" (Test-Path (Join-Path $Root "launch.py"))
Check "pipeline\\" (Test-Path (Join-Path $Root "pipeline"))
Check "command_center\\" (Test-Path (Join-Path $Root "command_center"))
Check "command_center\\web.py" (Test-Path (Join-Path $Root "command_center\\web.py"))
Check "command_center\\templates\\index.html" (Test-Path (Join-Path $Root "command_center\\templates\\index.html"))
Check "ui\\" (Test-Path (Join-Path $Root "ui"))
Check "config\\" (Test-Path (Join-Path $Root "config"))
Check "config\\pipeline_config.yaml" (Test-Path (Join-Path $Root "config\\pipeline_config.yaml"))
Check "config\\credentials.example.yaml" (Test-Path (Join-Path $Root "config\\credentials.example.yaml"))
Check "tests\\" (Test-Path (Join-Path $Root "tests"))
Check "scripts\\" (Test-Path (Join-Path $Root "scripts"))
Check "No populated .env" (-not (Test-Path (Join-Path $Root ".env"))) "(safe default)"

if (Get-Command python -ErrorAction SilentlyContinue) {
  & python .\run_pipeline.py --doctor
  Check "Model Lab Doctor" ($LASTEXITCODE -eq 0)
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
  Check "nvidia-smi" ($LASTEXITCODE -eq 0)
} else {
  Write-Host "[INFO] nvidia-smi not present (GPU is optional for the starter profile)" -ForegroundColor Yellow
}

Write-Host "Command Center launcher is present at root. Runtime API validation occurs when launched." -ForegroundColor Yellow

if ($fail -eq 0) { Write-Host "STATUS: GREEN" -ForegroundColor Green; exit 0 }
Write-Host "STATUS: FAIL ($fail)" -ForegroundColor Red; exit 1
