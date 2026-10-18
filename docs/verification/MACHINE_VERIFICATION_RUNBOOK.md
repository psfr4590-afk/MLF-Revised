# Machine Verification Runbook

## Purpose

Capture reproducible verification evidence for a specific ML-Framework candidate without exposing real credentials or accidentally starting an expensive training workload. This runbook is a procedure, not evidence that any gate has passed.

## Safety and setup

1. Use a disposable worktree or clean clone at the exact candidate commit. Do not clean, reset, or overwrite a developer's existing working tree.
2. Record `git rev-parse HEAD` and `git status --short` before testing.
3. Use a disposable test dataset and synthetic credentials only. Never include `.env`, tokens, private datasets, or model weights in the evidence bundle.
4. Record OS, architecture, Python version, dependency environment, and optional components actually installed (for example, Torch/CUDA, `psutil`, native compiler, and llama.cpp).
5. Do not run full native export or inference until the static checks pass and the target machine has enough resources.

## PowerShell evidence capture

Run from the candidate repository root in PowerShell 7 or a compatible PowerShell session:

```powershell
$ErrorActionPreference = "Continue"
$Evidence = Join-Path $env:TEMP ("ml-framework-evidence-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $Evidence -Force | Out-Null

git rev-parse HEAD 2>&1 | Tee-Object "$Evidence\commit.txt"
git status --short 2>&1 | Tee-Object "$Evidence\git-status.txt"
python --version 2>&1 | Tee-Object "$Evidence\python.txt"
python -c "import platform; print(platform.platform()); print(platform.machine())" 2>&1 | Tee-Object "$Evidence\platform.txt"
python -m pip freeze 2>&1 | Out-File "$Evidence\pip-freeze.txt"

python -m pytest -q tests/governance/test_requirement_acceptance.py tests/governance/test_operational_failure_modes.py tests/governance/test_runner_lifecycle.py tests/governance/test_runner_security.py 2>&1 | Tee-Object "$Evidence\governance-tests.txt"
$GovernanceExit = $LASTEXITCODE

python -m pytest -q 2>&1 | Tee-Object "$Evidence\full-suite.txt"
$FullSuiteExit = $LASTEXITCODE

@(
  "governance_exit=$GovernanceExit"
  "full_suite_exit=$FullSuiteExit"
) | Out-File "$Evidence\exit-codes.txt"

Write-Host "Evidence directory: $Evidence"
Write-Host "Governance test exit: $GovernanceExit"
Write-Host "Full suite exit: $FullSuiteExit"
```

Use the repository's documented environment and release-verification commands for the candidate. A failed dependency installation is a setup failure and must be recorded separately from a test failure. Preserve the actual exit codes; a log file existing is not proof that the command passed.

## Operational scenarios

- **Process-tree cancellation:** use a harmless fixture that starts a parent and child. Request cancellation and verify both exit within the documented timeout. Do not use a real training job as the first test.
- **Startup race:** repeat cancellation during process startup. Verify that no worker survives, duplicate launches are rejected, and the final run/stage state is terminal and consistent.
- **Restart recovery:** stop and restart the Command Center using a disposable run. Dead workers should transition to failure; unrelated live processes must not be adopted or killed.
- **PID reuse:** validate persisted process identity against process creation time or another run identity. Record any conservative-recovery limitation.
- **Redaction:** use synthetic values only. Check raw, JSON-escaped, Base64, URL-encoded, multiline, chunk-boundary, exception, event, and subprocess-output paths. Scan the relevant logs and serialized outputs without including the canary itself in the report.
- **Release gate:** from a clean clone, test missing RC flags and forced test failures. Each must return nonzero and must not create or retain a success report.
- **Clean clone:** follow the documented bootstrap procedure in a new directory. Record every command and its exit status.

## Release verification layers

Follow `docs/verification/VERIFICATION.md` and `docs/release/RELEASE_CHECKLIST.md` for the current repository gate.

- Static verification does not establish native GGUF export or inference.
- A green CI run does not prove behavior on a specific machine.
- The RC/native gate must be run on an appropriate target machine before claiming native export/inference readiness.
- Record actual invocation, candidate SHA, report path, exit code, and relevant artifact/inference result. Do not infer success from a workflow name or the existence of a report file.

## Evidence to return

Bundle the evidence directory only after checking it for secrets and unnecessary large/private artifacts. Include the exact SHA, clean/dirty status, OS/Python/toolchain details, commands and exit codes, test counts, stack traces, CI links, process parent/child observations, redaction results, and release-gate report/status. Keep synthetic credentials out of the bundle where possible; report only whether the scan found a canary.

## Interpretation

A passing governance test module alone does not establish release readiness. Keep unverified requirements explicitly unverified, and do not run destructive cleanup or release commands against production data.
