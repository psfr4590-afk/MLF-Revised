[CmdletBinding()]
param(
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Resolve the repository from the script location, not the caller's working directory.
# Windows PowerShell 5.1 can leave $PSCommandPath empty in some -File invocation paths,
# so fall back to MyInvocation.MyCommand.Path.
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $ScriptPath = $PSCommandPath
    if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
        $ScriptPath = $MyInvocation.MyCommand.Path
    }
    if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
        throw "Unable to determine the smoke script location. Run this file directly with powershell.exe -File."
    }
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)] [string]$Exe,
        [Parameter(Mandatory = $false)] [string[]]$Arguments = @()
    )

    Write-Host "`n> $Exe $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Arguments -join ' ')"
    }
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$Bootstrap = Join-Path $RepoRoot "bootstrap.py"
$Pipeline = Join-Path $RepoRoot "run_pipeline.py"
$Verifier = Join-Path $RepoRoot "scripts\verify_release.py"
$SmokeConfig = Join-Path $RepoRoot "config\pipeline_config.smoke.yaml"
$Venv = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Bootstrap)) { throw "bootstrap.py not found. Expected repository root: $RepoRoot" }
if (-not (Test-Path -LiteralPath $Pipeline)) { throw "run_pipeline.py not found. Expected repository root: $RepoRoot" }
if (-not (Test-Path -LiteralPath $SmokeConfig)) { throw "Smoke config not found: $SmokeConfig" }
if (-not (Test-Path -LiteralPath $Verifier)) { throw "Release verifier not found: $Verifier" }

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) is not on PATH. Install Python 3.12+ with the Python Launcher enabled."
}

$pythonVersion = & py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install it with: winget install Python.Python.3.12"
}

Write-Host "Using Python $pythonVersion" -ForegroundColor Green
Write-Host "Repository: $RepoRoot" -ForegroundColor Green

if (Test-Path -LiteralPath $VenvPython) {
    $venvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or $venvVersion -ne "3.12") {
        throw "Existing .venv is not Python 3.12. Remove '$Venv' and rerun this script so it can create the correct environment."
    }
} else {
    Invoke-Step "py" @("-3.12", "-m", "venv", $Venv)
}

# Do not self-upgrade pip here. The bootstrap owns dependency installation, and
# downloading the newest packaging tools makes a smoke test fail on slow networks.
Invoke-Step $VenvPython @($Bootstrap, "--install")
Invoke-Step $VenvPython @($Bootstrap, "--doctor")
Invoke-Step $VenvPython @($Pipeline, "--config", $SmokeConfig, "--no-resume")
Invoke-Step $VenvPython @($Verifier)

Write-Host "`nWindows smoke verification completed successfully." -ForegroundColor Green
