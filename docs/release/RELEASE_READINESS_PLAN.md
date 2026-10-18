# Model Lab Release Readiness

This document defines the production-readiness contract for `main`. It distinguishes repository readiness from evidence that can only be produced by executing the release gate. It does not cache live CI results.

## Current RC gate state

| Area | Current state | Evidence |
|---|---|---|
| Repository hardening | Implemented | current `main` source and merged implementation PRs |
| Linux CI definition | Present | `.github/workflows/ci.yml` |
| Windows CI definition | Present | `.github/workflows/ci.yml` |
| PowerShell contract | Present | `.github/workflows/ci.yml` |
| Dependency consistency | Gated | `pip check` in CI/security/release |
| Dependency vulnerability scan | Gated | `scripts/security_gate.py` |
| Secret scan | Gated | `scripts/security_gate.py` |
| Python compile | Gated | CI + `scripts/verify_release.py` |
| Ruff F lint | Gated | CI + `scripts/verify_release.py` |
| Automated test coverage | Reported by pytest-cov | `pyproject.toml` coverage configuration |
| Source provenance | Implemented | schema-3 source manifests (schema 2 remains readable) |
| Artifact source/config/code identity | Implemented | stage provenance contracts |
| Network-dependent release smoke | Removed | native release smoke uses local fixture |
| GGUF integrity | Gated | `scripts/verify_gguf.py` |
| Native llama.cpp inference | Required for RC/release | `scripts/verify_release.py --bootstrap-native` |
| Release dependency evidence | Implemented | freeze + SBOM |
| Current CI result | Execution-dependent | GitHub Actions for the exact commit under review |
| Termux native quantizer execution | Historical tool-level evidence | Android 16/aarch64; pinned llama.cpp revision; `llama-quantize --help` executed successfully on 2026-09-29 |
| Current native release result | **Pending execution** | complete `verify_release.py --bootstrap-native` and end-to-end export/inference remain required |
| Target-hardware validation | **Pending execution** | must be executed on intended deployment hardware |
| Production training/convergence | **Operational evidence** | separate from RC smoke |

## Release tag/version contract

`pyproject.toml` is version `1.3.0rc1` (PEP 440), matching the intended release-candidate tag `v1.3.0-rc.1`. The historical `v1.0.0-rc.1` tag is retained as a failed historical candidate and must not be reused or force-moved. Do not create or publish `v1.3.0-rc.1` until the required automated gates, native GGUF export/inference, target-hardware validation, and production-training evidence are complete. The release gate validates package version dynamically and the clean-clone gate verifies the exact approved commit instead of silently testing the default branch.

## Release commands

Static verification:

```text
python scripts/verify_release.py
```

Native smoke verification (native export and inference, but not full RC certification):

```text
python scripts/verify_release.py --bootstrap-native
```

Full release-candidate verification (also requires the UI/backend probe and clean-clone acceptance path):

```text
python scripts/verify_release.py --profile rc --bootstrap-native --ui-probe --clean-clone
```

Add `--verify-ollama` when Ollama deployment verification is part of the release target. The `--bootstrap-native` flag alone selects the `native-smoke` profile; it does not implicitly run the full RC profile.

The native gate is intentionally separate from ordinary CI because it requires the pinned llama.cpp toolchain and a target environment. A green CI run is necessary but is not sufficient for a native production release.

## Evidence retained by the release gate

- Python compile result
- Ruff result including `ui/`
- pytest result and coverage JSON
- `pip check` result
- security gate result
- resolved dependency freeze
- SBOM
- source manifest and source-definition hashes
- export manifest
- dataset and model cards
- GGUF SHA-256 validation
- llama.cpp inference result

The 2026-09-29 Termux evidence proves the native quantizer could be built and executed on Android 16/aarch64. It does not satisfy the full native release gate. The complete Model Lab gate remains intentionally separate and must be executed against the exact approved source, environment, and native dependency state.

## Release boundary

A release is approved only when every required automated gate is green and the native verification has passed on the intended target environment.

Missing evidence is a release-process failure, not an implicit pass.

Historical reports remain historical records and must not be presented as current release evidence.


## Open release blockers and closure evidence

The source-level training-estimate integration calibrates validation-pass and checkpoint-write overhead during preflight. If either calibration fails, the estimate records the missing measurement explicitly rather than representing compute-only time as a complete estimate. Configuration may override the measured values with `estimated_eval_seconds` and `estimated_checkpoint_seconds` when operators have reliable measurements from the target environment.

The following items cannot be closed by editing documentation or source code alone:

- [ ] Exact-commit Linux and Windows CI jobs pass, including tests and coverage.
- [ ] `python scripts/verify_release.py` passes on the approved commit.
- [ ] `python scripts/verify_release.py --bootstrap-native` completes GGUF export, integrity checks, and native inference on the intended target.
- [ ] Clean-clone acceptance succeeds for both Git clone and source ZIP installation paths.
- [ ] A documented training run shows stable execution and evaluation evidence sufficient to assess convergence; a short smoke run is not convergence proof.
- [ ] The release tag is created only after all required evidence above is retained.

No checkbox may be marked complete without an attached run artifact or exact-commit CI result. A code change can close a code defect; it cannot manufacture hardware or convergence evidence.
