# Unified Program Consolidation Record (v1.3.0rc1)

**Prepared:** 2026-10-17  
**Package identity:** `model-lab-framework` (`1.3.0rc1`)  
**Disposition:** Unified source-complete release candidate (`VERIFIED` on static, structural, governance, and offline unit/integration/acceptance gates; `BLOCKED` on target-host PyTorch training and native `llama.cpp` GGUF execution).

## 1. Input Artifact Inventory & Provenance

Ten input files were inspected and reconciled to produce this unified program tree:

| # | Uploaded File | Size (Bytes) | SHA-256 | Lineage Role |
|---|---|---:|---|---|
| 1 | `model-lab-framework-1.3.0rc1-production.zip` | 533,183 | `8a5b8282996f...` | Primary `1.3.0rc1` baseline (headless UI guards, resilient `DatasetStore._ids()`, non-git `security_gate.py` fallback, governance test suite). |
| 2 | `model-lab-framework-1.3.0rc1-production (1).zip` | 533,183 | `8a5b8282996f...` | Byte-identical duplicate of #1. |
| 3 | `model-lab-framework-1.3.0rc1-production (2).zip` | 680,349 | `ef8a75a0147d...` | Identical `ML-Framework-main/` tree as #1, plus an unrelated top-level `Harm.Less-source.zip` (147,050 B PWA) excluded from the unified ML-Framework build. |
| 4 | `ML-Framework-corrected-audited.zip` | 510,862 | `2d6764bff863...` | Audited `1.3.0rc1` snapshot (`docs/verification/ZIP_AUDIT_2026-10-15.md`: 256-symbol ByteLevel alphabet + SHA-256 fingerprint, NFKC removal, optional `sentence-transformers`/`faiss` fallback). |
| 5 | `ML-Framework-1.0.0.zip` | 505,680 | `7152f79877af...` | Version `1.0.0` snapshot containing two post-audit fixes (`semantic_dedup.py` `get_embedding_dimension` compatibility and `tests/test_tokenizer.py` `vocab_size=300` fix). |
| 6 | `ML-Framework-patched.zip` | 483,352 | `2e323635d103...` | Native toolchain & bootstrap hardening branch (`bootstrap.py` `third_party` exclusion, `bootstrap_llama_cpp.ps1`/`.sh` `-Cuda`/`--cuda` support, `llama-app` target build and runtime verification, stranger-clone secret/vendor checks). |
| 7 | `ML-Framework-corrected.zip` | 519,793 | `f459dbea1b9e...` | Cross-component defect-correction branch (`docs/development/CORRECTIONS.md`: `checkpoint_manifest_valid`, `select_export_checkpoint`, `pipeline/numerics.py` finite-loss guard, `pipeline/tokenizer/special_tokens.py` dynamic special-token ID resolver, full GGUF tensor-table/truncation/determinism verifier in `scripts/verify_gguf.py`, and contract tests `test_gguf_structure.py` + `test_integration_contracts.py`). |
| 8 | `ml_framework_tokenizer_hardening_bundle.zip` | 4,565 | `2de7eaf3727c...` | Standalone AST-verified ByteLevel pre-tokenizer overlay migration bundle (superseded by and verified against `scripts/patch_llama_cpp_tokenizer.py` and `tests/test_release_verification_contract.py`). |
| 9 | `mlframework_acceptance_complete.zip` | 29,015 | `852d3fd57301...` | Local offline acceptance kit (`config/pipeline_config.a1f.yaml`, `config/dataset_groups.a1f.yaml`, `scripts/run_a1f_acceptance.py`, `tests/acceptance/test_a1..a5*.py`, and `tests/fixtures/filesystem_corpus/*`). |
| 10 | `ML-Framework-source_manifest-2026-10-11_23-43-09.json` | 5,034 | `9d9424925c61...` | Schema-3 live smoke crawl `source_manifest.json` (`run_id: 951aaf350b3d4069899eace685ddf781`), preserved at `docs/release/source_manifest_smoke_2026-10-11.json`. |

## 2. Reconciled Cross-Branch Changes & Defect Fixes

1. **Orchestrator (`pipeline/orchestrator.py`)**:
   - Preserved the audited 256-symbol ByteLevel alphabet size and SHA-256 fingerprint in `_stage_tokenize` (`1.3.0rc1` / `corrected-audited`).
   - Integrated `self._selected_checkpoint` pass-through from `_stage_train` to `_stage_export` (`corrected`) so the exporter always receives the exact validated checkpoint chosen by the training stage.
2. **Command Center Store & Web (`command_center/store.py`, `command_center/web.py`)**:
   - Preserved resilient dataset directory discovery (`_ids()`) and idempotent `ensure_seed_datasets()` (`1.3.0rc1`).
   - Integrated `checkpoint_manifest_valid` (accepting schema >= 2 with `kind == "checkpoint"`) and `ckpt_final.pt` + `ckpt_final_*.pt` detection (`corrected`).
   - Removed deprecated `@app.on_event("startup")` from `command_center/web.py` (which caused `tests/test_command_center_lifespan.py::test_command_center_has_no_deprecated_startup_event` to fail in `1.3.0rc1`) and bound `lifespan` cleanly onto `app.router.lifespan_context`.
3. **Checkpoint Integrity, Special Tokens, Numeric Safety & GGUF Verification**:
   - Added `pipeline/numerics.py` (`NonFiniteLossError`, `require_finite`) and wired finite-loss enforcement into `pipeline/trainer/train.py`.
   - Added `pipeline/tokenizer/special_tokens.py` (`special_token_ids`) and wired dynamic special-token resolution into `scripts/export_gguf.py`, `scripts/verify_release.py`, and `mlframework.py`.
   - Upgraded `scripts/verify_gguf.py` with bounds-checked `_Reader`, full GGUF tensor-table and byte-completeness checks (`read_gguf_structure`, `validate_llama_tensors`), deterministic `--temp 0 --seed` repeat-run verification, and `--evidence-dir` output.
   - Added `tests/test_gguf_structure.py` and `tests/test_integration_contracts.py`.
4. **Bootstrap & Native Toolchain (`bootstrap.py`, `scripts/bootstrap_llama_cpp.ps1`, `scripts/bootstrap_llama_cpp.sh`)**:
   - Added `third_party` exclusion to `_source_import_dependency_issues` and stranger-clone checks (`patched`).
   - Added `-Cuda` / `--cuda` opt-in, non-destructive checkout verification, `llama-app` target compilation, and `--help` / `--version` binary smoke checks (`patched`).
5. **Local Acceptance Kit (`config/*.a1f.yaml`, `scripts/run_a1f_acceptance.py`, `tests/acceptance/*`, `tests/fixtures/filesystem_corpus/*`)**:
   - Integrated all A1-F, A2, A4, and A5 acceptance tests, offline corpus fixtures, and runner script (`acceptance_complete`), with documentation at `docs/verification/ACCEPTANCE_KIT_A1F.md`.
6. **Governance v1.1.0 Catalog & Validators (`governance/v11/`)**:
   - Populated `governance/v11/` with byte-identical v1.0.0 baseline files under `governance/v11/history/`, verified checksum manifests (`SHA256SUMS-HISTORY.txt`, `SHA256SUMS.txt`), the 64-requirement catalog (`requirements_catalog.csv`), traceability matrix (`traceability_matrix.csv`), test catalog (`test_catalog.csv`), and executable validators (`validate_governance.py`, `check_test_coverage.py`).
7. **Torch-Free Import Isolation**:
   - Guarded top-level `import torch` in `pipeline/trainer/model.py`, `pipeline/trainer/train.py`, `pipeline/preflight.py`, `scripts/export_gguf.py`, and `scripts/evaluate_model.py` so pure-Python configuration, sizing, provenance, and manifest verification functions import and pass tests in environments where PyTorch is not yet installed.
