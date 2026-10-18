# Repository Mapping Addendum (non-governing companion)

**Repository:** https://github.com/psfr4590-afk/ML-Framework.git  
**Default branch:** `main`  
**Reviewed source reference:** `f06bbedb55e252cd7033c615733c5d2554a7d0aa`  
**Scope:** selected source/config/test/document files retrieved from GitHub; no clone or test execution was performed during this mapping pass.

## Confirmed repository facts

- `pyproject.toml` identifies project version `1.3.0rc1`, Python `>=3.11,<3.15`, and console entry points `mlab`, `mlframework`, `mlab-command-center`, and `mlab-launch`.
- `README.md` documents Windows/Linux/macOS/headless Termux intent and a single localhost browser Command Center; `launch.py` and `ui/app.py` are compatibility entry points.
- `run_pipeline.py` declares stage order `crawl`, `clean`, `dedup`, `weight`, `tokenize`, `shard`, `train`, `export` and default config `config/pipeline_config.yaml`.
- `config/pipeline_config.yaml` starter config has export disabled, semantic dedup enabled in `auto` mode with model download disabled, and a bounded two-step 85M training profile. This is a smoke profile, not a model quality claim.
- `pipeline/types.py` defines the canonical `Document` dataclass; `pipeline/contracts/run.py` aliases that type rather than defining a second schema.
- `pipeline/experiment_db.py` declares SQLite schema version 2 and tables for experiments, runs, datasets, sources, stages, artifacts, checkpoints, metrics, evaluations, hardware, configs, warnings, errors, and runtime estimates.
- `command_center/web.py` defines loopback-only API middleware, request-size checks, Origin filtering for mutations, and required `x-m2s-command-center: 1` mutation header.
- `command_center/store.py` uses dataset directories `dataset_NNN`, a positive integer dataset ID, process lock, and temp-file plus `os.replace` persistence for `dataset.json`.
- `scripts/verify_release.py` provides separate static and native release verification; `docs/verification/VERIFICATION.md` documents the full RC command and report path.
- Inspected tests include structural assertions in `tests/model_lab/test_model_lab_structure.py` and regression tests in `tests/test_audit_regressions.py` for provenance resume checks, run finalization ownership, DB connection closing, canonical dedup naming, release gate, dependency and config contracts.

## Findings requiring explicit handling

1. Application version `1.3.0rc1` is not the same namespace as governance specification `v1.1.0`. Release tags/metadata/docs must align before any release claim.
2. The source declares broad platform intent, but this review did not find candidate-bound clean-install results for every OS/Python/GPU/native combination. Support matrix stays `NOT_VERIFIED`.
3. The native RC gate exists but was not run in this audit. Existence of a gate is not evidence it passed.
4. Several requirement families are represented by structural/regression tests; individual functional/security/recovery acceptance tests remain to be mapped or added. The traceability register uses explicit `ADD REQUIRED` gaps rather than inventing test functions.
5. Dataset card, model card, evaluation threshold, resource bounds, retention, and recovery targets require a formal approved decision or must remain blocked wherever a release claim depends on them.
6. `main` is mutable. All real verification evidence must pin the commit actually tested; the reviewed snapshot reference above is not a claim that it remains current HEAD.

## Existing test functions inspected

`tests/model_lab/test_model_lab_structure.py` includes:
- `test_package_identity_is_model_lab`
- `test_root_launcher_is_the_canonical_command_center_launcher`
- `test_compatibility_launcher_delegates_to_the_same_interface`
- `test_command_center_is_the_only_browser_operator_surface`
- `test_pipeline_stage_contract_is_complete`
- `test_all_stage_routes_are_exposed_by_command_center`
- `test_stop_route_exists`
- `test_dataset_routes_cover_list_create_get_ingest`
- `test_group_route_exists`
- `test_system_route_exists`
- `test_credential_routes_cover_list_set_test_delete`
- `test_credential_presets_are_exactly_the_four_required_slots`
- `test_documentation_contains_one_operator_interface`
- `test_no_second_desktop_launcher_is_advertised`
- `test_no_common_secret_literals_are_committed`
- `test_release_docs_exist`
- `test_machine_verification_suite_is_present`
- `test_traceability_document_exists`

`tests/test_audit_regressions.py` includes:
- `test_semantic_dedup_dependencies_are_required_everywhere`
- `test_every_canonical_dataset_group_has_meaningful_high_quality_web_seeds`
- `test_starter_requires_semantic_dedup_and_uses_local_model_by_default`
- `test_resume_skip_requires_exact_artifact_provenance`
- `test_resume_skip_rejects_byte_corruption`
- `test_run_finalization_is_owned_by_the_pipeline_boundary`
- `test_experiment_db_exposes_explicit_close`
- `test_dedup_stage_uses_one_canonical_stage_name`
- `test_release_verifier_contains_network_free_export_and_inference_gate`
- `test_runtime_and_documentation_contracts_match_source`

The list above is not exhaustive of the repository's test suite and does not represent executed results. The reviewed source reference is a commit whose content was inspected; this audit does not assert that it is the current HEAD of `main`.
