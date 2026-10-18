# Model Lab Verification

Model Lab verification is divided into automated contract tests, Python compilation, non-destructive environment checks, native artifact verification, and localhost browser Command Center connectivity checks.

## Static verification

Run the repository-side gate:

```bash
python scripts/verify_release.py
```

This runs compilation, Ruff, pytest with coverage reporting, the project/runtime doctor, and the security gate. It does not claim native GGUF export or inference.

## Full native verification

Run the native gate on the machine intended to perform export/inference:

```bash
python scripts/verify_release.py --bootstrap-native --profile rc --ui-probe --clean-clone
```

The native gate exercises the deterministic local fixture, export integrity, native llama.cpp inference, UI/backend connectivity, and the documented clean-clone path. `--profile rc` enforces the full RC acceptance criteria and requires `--ui-probe` and `--clean-clone`. Its machine-readable result is written to `release-evidence/release_report.json`.

## Windows checks

On the target Windows machine:

```powershell
python -m pytest -q .\\tests\\model_lab\\test_machine_environment.py
powershell -ExecutionPolicy Bypass -File .\\scripts\\run_release_verification.ps1 -IncludeMachineChecks
```

The machine suite verifies supported Python, required executables, CUDA observability when available, and Command Center health. Legacy Tkinter/display checks are not release gates.

Machine-specific evidence must not be presented as a universal guarantee for other hardware.

The release checklist is maintained in `docs/release/RELEASE_CHECKLIST.md` and the traceability checklist in `docs/verification/VERIFICATION_CHECKLIST.md`.
