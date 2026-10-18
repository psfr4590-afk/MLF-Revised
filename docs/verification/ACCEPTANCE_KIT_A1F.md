# ML-Framework local acceptance kit: A1-F, A2, A4, A5

This is an additive local test/config package for `psfr4590-afk/ML-Framework`. It does not create branches, commits, PRs, issues, or any other GitHub-side changes. All tests and pipeline artifacts are local.

## Scope

- **A1-F**: filesystem fixture ingestion, source file hashes, extension/size bounds, explicit source isolation, optional full pipeline integration through training and GGUF export.
- **A2**: stable configuration identities, artifact manifest hashes, and tamper detection.
- **A4**: training compatibility identity checks that separate continuation controls (`resume`, `total_steps`) from model compatibility settings. Full crash/resume continuation is a separate integration gate and is not claimed green by the unit tests alone.
- **A5**: exporter rejects legacy/incomplete training provenance. The full A1-F integration test additionally requires a real GGUF output.
- Existing offline semantic deduplication, shard integrity, and training provenance lineage regression tests are included in the test command.

The project currently exposes `85M` as its smallest named model preset. This kit does not silently call that model “tiny”; the profile is a bounded CPU smoke run, not proof that the requested tiny-model profile exists.

## Install into a local checkout

Copy/merge these directories into the repository root:

- `tests/`
- `config/dataset_groups.a1f.yaml`
- `config/pipeline_config.a1f.yaml`
- `scripts/run_a1f_acceptance.py`

Do not overwrite existing files. Use Python 3.11 through 3.14 and install test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## 1. Run fast acceptance/regression tests

From the repository root:

```bash
python scripts/run_a1f_acceptance.py
```

The runner writes `.acceptance/acceptance_report.json` with the commands, output, exit status, and timestamps. This fast gate intentionally does not pretend it ran model training or GGUF export.

## 2. Run the full local pipeline gate

Prerequisites:

1. `third_party/llama.cpp` exists and contains a usable llama.cpp GGUF converter compatible with this project's exporter.
2. PyTorch and all dependencies from `requirements-dev.txt` are installed.
3. Enough RAM and disk are available for the project's smallest currently named model preset, `85M`.

Then run:

**Linux/macOS**
```bash
MLFRAMEWORK_RUN_FULL_A1F=1 python -m pytest -q tests/acceptance/test_a1_filesystem_end_to_end.py
```

**PowerShell**
```powershell
$env:MLFRAMEWORK_RUN_FULL_A1F = "1"
python -m pytest -q tests/acceptance/test_a1_filesystem_end_to_end.py
```

The config enables only `local_files`; all network source kinds are explicitly false both globally and in the selected dataset group. Semantic deduplication uses the deterministic fallback and model downloads are disabled. Output is redirected to `.acceptance/output` and `.acceptance/scratch`.

The integration test blocks Python socket connection entry points during pipeline execution, then requires stage statuses, source manifest, tokenizer, shard manifest, checkpoint, and at least one `.gguf` file. A missing converter or any stage failure is a failure, not a warning to wave away.

## 3. Manual commands

```bash
python run_pipeline.py --doctor
python run_pipeline.py --list-stages
python -m pytest -q tests/acceptance tests/test_semantic_dedup_offline.py tests/test_shard_integrity.py tests/test_training_provenance_lineage.py
```

Full run:
```bash
python run_pipeline.py --config config/pipeline_config.a1f.yaml --stages all --no-resume
```

## Acceptance limits that remain explicit

- A1-F's full run is only PASS if the real end-to-end integration test completes including GGUF export.
- The kit does not add a new tiny architecture to production code because this is a test kit, and you asked that the GitHub repository remain untouched. A true tiny-model profile needs a separately reviewed local code change.
- A2 unit tests validate the manifest/hash helpers. For a full run, inspect `.acceptance/output/runs/<run-id>/run_manifest.json`, `.acceptance/output/source_manifest.json`, and all adjacent artifact manifests.
- A4 unit tests validate the checkpoint-compatibility identity boundary. A true interrupted-training resume acceptance must interrupt a live training process after a checkpoint, restart with the same compatible config and resume enabled, then verify continued step numbering and unchanged upstream artifact identities. Do not mark A4 full PASS based only on these unit tests.
- Reproducibility requires two clean runs with identical source bytes/config/seed and a canonical comparison of artifact hashes. Run IDs and timestamps are volatile and must be excluded; model checkpoint hashes may differ across hardware/runtime versions and should be reported rather than silently normalized.

## Evidence to retain

Keep `.acceptance/acceptance_report.json`, the run manifest, source manifest, dataset report, tokenizer/shard manifests, checkpoint plus checkpoint manifest, export lineage/report, GGUF file hash, and the terminal output. The GitHub repository is not changed by this package.
