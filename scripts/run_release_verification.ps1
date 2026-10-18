param([switch]$IncludeMachineChecks,[switch]$NoDoctor)
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-Checked {
  param([Parameter(Mandatory=$true)][string]$Exe,[Parameter(Mandatory=$false)][string[]]$Arguments=@())
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE: $Exe $($Arguments -join ' ')"
  }
}

if(-not $NoDoctor){Invoke-Checked "python" @(".\run_pipeline.py","--doctor")}
Invoke-Checked "python" @("-m","pytest","-q")
Invoke-Checked "python" @("-m","compileall","-q",".\ui",".\command_center",".\pipeline",".\run_pipeline.py",".\run_command_center.py",".\launch.py")
if($IncludeMachineChecks){Invoke-Checked "python" @("-m","pytest","-q",".\tests\model_lab\test_machine_environment.py")}
