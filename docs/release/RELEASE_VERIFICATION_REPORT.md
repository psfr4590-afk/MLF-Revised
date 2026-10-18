# Historical Release Verification Report

> **Historical record.** This report documents the repository state and environment verification performed on **2026-08-26**. Its test counts and environment findings are not the current release status. For current status, use the GitHub Actions CI result and the current release-gate documentation.

Date: 2026-08-26

## Code-level verification

- Python compilation: PASS
- Automated tests: **82 passed, 11 skipped**
- Checkpoint integrity round-trip: PASS
- Export tensor mapping tests: PASS
- Project-root launch behavior: PASS
- Dataset artifact isolation tests: PASS

## Native/environment verification

The build environment at the time of this report did **not** contain the required `tokenizers` package or the pinned llama.cpp checkout, and had no CUDA runtime. Therefore the strict release gate correctly remained **BLOCKED** in that environment. This was intentional. The project did not claim a native GGUF conversion it had not executed.

Windows:

```powershell
python -m pip install -r requirements.txt
.\scripts\bootstrap_llama_cpp.ps1
python run_pipeline.py --doctor
python scripts/verify_release.py
```

Termux / Android:

```bash
python -m pip install -r requirements.txt
bash scripts/bootstrap_llama_cpp.sh
python run_pipeline.py --doctor
python scripts/verify_release.py
```

The llama.cpp integration was pinned to upstream release `b10516` / commit `b95502b`. The bootstrap script verifies the resolved Git commit before building.

A final release required a real reduced end-to-end run through tokenization, sharding, training, GGUF conversion, quantization, and llama.cpp inference.

## Why this file remains

This document is retained as an audit trail of the August 26 verification effort. It should not be edited to reflect later test counts. New verification results belong in a new dated report or in the repository's current CI/release documentation.
