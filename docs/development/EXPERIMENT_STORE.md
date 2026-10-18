# SQLite experiment store

Model Lab now keeps a structured SQLite source of truth alongside the existing raw run artifacts. Phase 6.1 established the schema and automatic synchronization; Phase 6.2-6.4 now project retained dataset/source reports, live metrics, checkpoint manifests, preflight/runtime evidence, validation metrics, and historical run manifests.

## Contract

`output/experiment.db` records experiments, runs, datasets, sources, source statistics, stages, artifacts, checkpoints, metrics, evaluations, hardware, configuration identities, warnings, errors, and runtime estimates.

The JSON run manifest and raw logs remain forensic evidence. The SQLite store does not replace them, and the system does not automatically delete experiments or raw evidence.

The durable identity chain is:

`experiment -> run -> dataset/config/hardware -> stages -> artifacts`

Artifacts retain SHA-256 values and paths so the structured record can be checked against retained evidence.

## Querying

```python
from pathlib import Path
from pipeline.experiment_db import ExperimentDB

db = ExperimentDB(Path("output/experiment.db"))
print(db.search_runs(status="PASS"))
print(db.get_run("run-id"))
print(db.lineage("run-id"))
print(db.compare_runs("run-a", "run-b"))
db.close()
```

The store is created automatically when a pipeline run tracker is initialized. Every run-manifest update is synchronized into SQLite, so an interrupted run still leaves queryable state.

## Evidence boundary

SQLite describes what Model Lab knows about the run. Raw manifests, logs, checkpoints, dataset artifacts, and other retained files preserve what the system actually observed. Neither silently replaces the other.

This is an observability and reproducibility store, not a claim that a dataset, model, or training run is production-ready.


## Historical import

Existing retained runs can be projected without modifying their raw evidence:

    python scripts/import_experiment_runs.py --output-dir output

Use `--run-id` to import one run. The importer is idempotent and never deletes or rewrites retained run manifests, logs, checkpoints, or dataset artifacts.

## Evidence projected by Phase 6.2-6.4

- Dataset report counts and SHA-256 identity.
- Phase 4 source retrieval records and normalized request/HTTP/document statistics.
- Stage metrics and artifact hashes.
- `logs/metrics.jsonl` step-level training telemetry.
- Checkpoint integrity manifests, steps, validation loss, best/final state.
- `preflight_report.json` estimates and measured throughput.
- Training initial estimates versus actual duration when available.
- Validation metrics as structured evaluations.
- Warnings and errors.
- Hardware and independent configuration identities.

The SQLite record is therefore useful after the raw log is closed: a run can be reopened by ID, traced through dataset/source/stage/artifact lineage, compared with another run, and inspected for checkpoints, metrics, evaluations, hardware, warnings, errors, and runtime estimates.

## Phase 6 boundary

Phase 6 is complete when the integration tests and CI verify this projection against the actual run-manifest/evidence contracts. It does not claim that a production training experiment has succeeded. That remains an execution-evidence milestone for the project as a whole.
