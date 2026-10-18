# Model Lab Release Candidate / Production Release Checklist

A release candidate is a source-tree candidate plus reproducible evidence. A production release requires the additional target-environment gates. This checklist is intentionally reset for each release candidate; checked items are not permanent claims about the current `main` branch.

## 1. Repository and automated gates

- [ ] Exact release commit identified and frozen before tagging
- [ ] Release tag exactly matches `pyproject.toml` version (`v<project-version>` or the approved prerelease suffix)
- [ ] Historical mismatched tags are retained as failed candidates and are not force-moved
- [ ] Linux CI passes on the exact release commit
- [ ] Windows CI passes on the exact release commit
- [ ] Windows PowerShell bootstrap contract passes
- [ ] `python -m compileall -q .` passes in CI
- [ ] `python -m pytest -q` plus `python -m coverage report --fail-under=60` passes in CI
- [ ] Isolated clean-install CI job installs the package from source and passes doctor/entrypoint checks
- [ ] `python run_pipeline.py --doctor` has no required failures
- [ ] Security gate passes: secret scan, `pip check`, `pip-audit`
- [ ] CodeQL analysis passes on the exact release commit
- [ ] No unintended generated artifacts, credentials, caches, or native build products are tracked

## 2. Native release gate

- [ ] Pinned llama.cpp bootstrap completed and the expected revision is recorded
- [ ] `convert_hf_to_gguf.py` exists in the pinned checkout
- [ ] `llama-quantize` exists when quantized export is enabled
- [ ] Deterministic network-free release fixture completes
- [ ] Tokenizer and shard contracts pass
- [ ] Reduced training run writes a checkpoint and integrity manifest
- [ ] Checkpoint reload succeeds
- [ ] GGUF F16 export succeeds
- [ ] Requested quantized GGUF export succeeds
- [ ] Export manifest hashes match files on disk
- [ ] Dataset and model cards are generated and hashed
- [ ] `scripts/verify_gguf.py` passes
- [ ] llama.cpp native inference validation passes

## 3. Provenance, security, and reproducibility

- [ ] Every production source has a retained source manifest with identifier, retrieval time, revision, license/usage terms, and raw-source SHA-256 where available
- [ ] Attribution and redistribution requirements have been reviewed before publishing a dataset or model
- [ ] Sources without established training rights are excluded
- [ ] Crawler boundary/security tests pass
- [ ] Untrusted document/archive parsing limits are enforced
- [ ] Credential storage and redaction controls are verified
- [ ] Command-center localhost binding and access controls are verified
- [ ] Dependency vulnerability/license scanning is complete for the release environment
- [ ] Dependency freeze and SBOM are retained
- [ ] Dependency authority is `pyproject.toml`; adapter requirements files contain no duplicated runtime dependency declarations
- [ ] Release artifact receives a GitHub artifact attestation

## 4. Artifact lineage

- [ ] Full Git commit SHA is recorded
- [ ] Pipeline configuration SHA-256 is recorded
- [ ] Source manifest SHA-256 is recorded
- [ ] Shard manifest SHA-256 is recorded
- [ ] Model configuration and random seeds are recorded
- [ ] Hardware and driver information are recorded for training runs
- [ ] Deterministic-mode settings are recorded when deterministic output is promised
- [ ] Final GGUF SHA-256 is recorded

## 5. Production-only gates

These are not satisfied by the bounded RC smoke test:

- [ ] Clean-clone acceptance is performed against the exact approved release commit
- [ ] Command Center UI/backend probe passes against the exact approved release commit
- [ ] Target deployment hardware passes native export/inference
- [ ] Intended production training run completes successfully
- [ ] Training loss/convergence and evaluation evidence are recorded
- [ ] Production dataset provenance/rights review is complete
- [ ] Release artifact is rebuilt from the exact approved source/config/dependency state
- [ ] Native toolchain evidence records pinned llama.cpp commit, converter hash, native binary hashes, and CMake version
- [ ] GitHub Release is published from the exact matching version tag with verified release assets

Every completed gate should include its execution environment, result, and relevant artifact or log evidence. Historical reports do not satisfy current gates.
