# Model Lab 1.3.0 Delivery Notes

> **Historical document.** This file records the contents and verification boundary of the original delivery archive. It is not current release evidence and does not override the code, tests, or documentation on `main`.

## Historical archive contents

The archive was based on the production-trusted r2 baseline and included the following work:

- Global semantic-dedup decision scope across embedding batches.
- Higher-weight semantic duplicates replacing lower-weight representatives before final emission.
- Persisted crawler telemetry under each dataset scratch directory.
- FastAPI endpoints for crawler statistics, domain signal telemetry, and log tail.
- Tk application styling with a safe fallback for minimal Tk installations.
- `command_center.app` as a compatibility export of the authoritative FastAPI application.
- Regression coverage for cross-buffer semantic deduplication.

## Historical verification boundary

The archive verification recorded Python compilation, automated tests, a FastAPI command-center smoke test, and crawler security smoke coverage. The test count recorded here is historical and must not be used as the current repository test count.

Machine-dependent release gates were intentionally not certified by that archive. Those gates included tokenizer integration, CUDA training, llama.cpp conversion and quantization, and runtime inference. Current release status must be established from the CI results and target-machine verification documented for the current `main` branch.

## Current-source rule

Use the repository `main` branch, its CI results, and current release documentation for present-state claims. Do not copy historical test counts, environment limitations, or release conclusions from this document into current documentation.
