# Model Lab Architecture

Model Lab is a local-first end-to-end training framework. The public repository contains the source, configuration, tests, and release tooling. Runtime datasets, checkpoints, logs, scratch state, and native build products are created outside version control.

## System flow

```text
seed URLs / dataset groups
        |
        v
     CRAWL  ---- source scoring / URL security / domain signals
        |
        v
     CLEAN  ---- HTML + Unicode normalization + quality/refusal filtering
        |
        v
   SEMANTIC DEDUP ---- exact guards + optional embedding acceleration
        |
        v
     WEIGHT  ---- domain/content weighting and sampling policy
        |
        v
    TOKENIZE ---- deterministic tokenizer artifacts
        |
        v
     SHARD   ---- validated binary training shards + manifest
        |
        v
     TRAIN   ---- hardware-aware bounded training + checkpoints
        |
        v
    EXPORT   ---- HF checkpoint -> GGUF -> optional quantization
```

## Repository layers

### `pipeline/`
The production data and training path. Stage implementations are deliberately separated so each stage can be tested and resumed independently.

- `crawler/`: source adapters, URL/security policy, and source scoring
- `cleaner/` and `filtering/`: text normalization and quality filtering
- `embedder/` and `dedupe/`: semantic and compatibility deduplication
- `weighter/`: corpus balancing policy
- `tokenizer/`: tokenizer training and reload
- `shardwriter/`: deterministic binary shard generation
- `trainer/`: model definition and training loop
- `integrity.py`: artifact validation, hashing, manifests, and atomic writes
- `model_sizer.py`: hardware-aware training profile selection
- `orchestrator.py`: stage sequencing, resume boundaries, and artifact chaining
- `contracts/`: stable compatibility namespace for shared pipeline data contracts

`pipeline.types.Document` is the canonical document schema. `pipeline.contracts.run.Document` is intentionally a compatibility alias so the project has one schema instead of two subtly different ones.

### `command_center/`
The localhost control plane and sole operator surface. It owns dataset sessions, credentials, process lifecycle, API routes, crawler telemetry, and browser-facing operational state. It reads and writes the same backend state used by CLI execution.

### `ui/`
Legacy compatibility-only code retained outside the distributable package. `ui/app.py` delegates to the browser Command Center; the legacy Tk screens are not the supported operator interface and are not part of the release operator contract.

### `config/`
The canonical configuration surface. Dataset groups, seeds, source weights, cleaner policy, pipeline settings, credentials templates, and smoke-test settings live here. Runtime credentials are never intended to be committed.

### `scripts/`
Release and environment tooling: native llama.cpp bootstrap, GGUF export, hardware recommendation, environment reconciliation, preflight, runtime helpers, and release verification.

### `tests/`
Tests are split by purpose rather than implementation convenience:

- root-level tests cover contracts and production behavior
- `tests/model_lab/` covers the Model Lab release surface, service behavior, structure, and machine-environment boundaries

## Runtime boundaries

Model Lab is intentionally explicit about what can be verified in generic CI and what depends on the target host.

CI can verify Python compilation, configuration contracts, service behavior, artifact logic, and tests. CUDA availability, native llama.cpp builds, network access, and human-visible Tk behavior are target-machine concerns and must not be represented as unconditional CI passes.

## Dataset isolation

A dataset session lives under `datasets/dataset_NNN/`. Its output and scratch paths are isolated from other sessions. Resume logic validates persisted artifacts before skipping a stage. Corrupt or incomplete artifacts are rebuilt instead of being silently trusted.

## Extension rules

1. Add new data sources as crawler adapters. Do not duplicate crawling logic in the UI.
2. Add new cleaning policies through the cleaner/filtering layer and configuration.
3. Preserve `Document` compatibility when adding fields. Avoid creating parallel document dataclasses.
4. Treat artifact manifests and hashes as part of the stage contract.
5. Keep optional heavy dependencies lazy so doctor, tests, and lightweight stages remain usable.
6. Never commit runtime credentials, datasets, checkpoints, caches, or generated logs.

## Release principle

The release is production-quality when the source tree is reproducible, the pipeline stages have explicit contracts, automated tests pass, environment-dependent checks are honestly reported, and a fresh clone can discover the supported paths without relying on private development history.
