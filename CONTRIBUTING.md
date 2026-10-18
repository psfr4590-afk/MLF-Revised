# Contributing

ML-Framework is intentionally structured around a small number of canonical pipeline stages. Contributions should improve the existing architecture rather than introduce parallel implementations of the same behavior.

## Development principles

1. Keep pipeline behavior deterministic and resumable where practical.
2. Preserve artifact integrity checks and atomic writes.
3. Keep credentials and generated training artifacts out of source control.
4. Prefer contract tests for stage boundaries and integration tests for cross-stage behavior.
5. Do not silently turn a failed hardware, dependency, provenance, or release check into a pass.
6. Keep user-facing commands and documentation aligned with the actual installed entry points.

## Validation

Before opening a pull request, run the non-destructive checks from a clean environment when practical:

```powershell
python .\\run_pipeline.py --doctor
python -m pytest -q
python .\\scripts\\verify_release.py
```

For the full native release gate, use:

```powershell
python .\\scripts\\verify_release.py --profile rc --bootstrap-native --ui-probe --clean-clone
```

Target-machine checks should be run on the hardware that will actually execute training or export. Native and production evidence must be recorded separately from static CI results.

## Pull requests

Keep PRs focused and explain the contract being changed. Include tests for behavior changes and update the relevant documentation when commands, configuration, release boundaries, or supported environments change.

Do not commit datasets, checkpoints, model artifacts, caches, native build products, populated credential files, or machine-specific output. Use the repository's ignored runtime directories for generated evidence.

Before merge, confirm the exact PR head passes the required CI, security, and CodeQL checks. After merge, treat `main` as the source of truth and close or delete superseded release branches where repository tooling permits it.
