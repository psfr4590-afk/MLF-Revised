[CmdletBinding()]
param(
    [switch]$Native
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Python {
    $candidates = @(
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "py"; Args = @("-3.14") },
        @{ Command = "py"; Args = @("-3.13") },
        @{ Command = "py"; Args = @("-3.12") },
        @{ Command = "py"; Args = @("-3.11") },
        @{ Command = "python"; Args = @() },
        @{ Command = "python3"; Args = @() }
    )
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate.Command -ErrorAction SilentlyContinue
        if ($cmd) {
            try {
                $versionText = & $cmd.Source @($candidate.Args) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($LASTEXITCODE -eq 0 -and $versionText -match '^3\.(11|12|13|14)$') {
                    return @{ Command = $cmd.Source; Args = $candidate.Args }
                }
            } catch {}
        }
    }
    return $null
}

$Python = Find-Python

if (-not $Python) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Error @"
Python 3.11-3.14 is required, but no supported interpreter was found.
Install Python from https://www.python.org/downloads/ or install a package manager that provides winget, then run this script again.
"@
        exit 2
    }

    Write-Host "No supported Python was found. Installing Python 3.13 with winget..."
    & $winget.Source install --id Python.Python.3.13 -e --scope user --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python installation failed. Install Python 3.11-3.14 manually, then run .\bootstrap_windows.ps1 again."
        exit $LASTEXITCODE
    }

    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $Python = Find-Python
    if (-not $Python) {
        Write-Error "Python was installed, but this PowerShell session cannot find it yet. Close PowerShell, open a new PowerShell window in this folder, and run .\bootstrap_windows.ps1 again."
        exit 2
    }
}

$version = & $Python.Command @($Python.Args) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "Using Python $version"

& $Python.Command @($Python.Args) (Join-Path $Root "bootstrap.py") "--install"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Project dependency installation failed. Re-run this script after resolving the reported pip/network error."
    exit $LASTEXITCODE
}

& $Python.Command @($Python.Args) (Join-Path $Root "bootstrap.py") "--doctor"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Bootstrap doctor reported a failure. Fix the listed prerequisite or dependency and run this script again."
    exit $LASTEXITCODE
}

if ($Native) {
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmake) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $winget) {
            Write-Error "Native export requires CMake. Install CMake, then rerun .\bootstrap_windows.ps1 -Native."
            exit 2
        }
        Write-Host "CMake was not found. Installing CMake with winget..."
        & $winget.Source install --id Kitware.CMake -e --scope user --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Error "CMake installation failed."
            exit $LASTEXITCODE
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    }

    & $Python.Command @($Python.Args) (Join-Path $Root "bootstrap.py") "--ensure-llamacpp"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Native llama.cpp setup failed. The basic Python installation is complete, but native export is not ready."
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "ML-Framework first-boot setup complete."
Write-Host "Next: python .\mlframework.py smoke"
if (-not $Native) {
    Write-Host "Native export: python .\bootstrap.py --ensure-llamacpp"
}
