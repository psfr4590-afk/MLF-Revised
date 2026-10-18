# ML-Framework Intended Architecture

**Document ID:** MLF-GOV-004  
**Version:** 1.0.0  
**Status:** Approved architecture baseline  
**Prepared:** 2026-10-11  
**Approval:** Approved by product owner on 2026-10-11

## 1. Purpose

This document defines the intended architecture and its governing boundaries before implementation decisions are made. It uses lessons from development to prevent recurring classes of failure; it does not claim that the current code is already organized this way.

## 2. Architectural goals

The architecture should provide:
- One coherent browser Command Center.
- Clear separation between presentation, application orchestration, domain workflows, execution adapters, and persistence.
- Durable and traceable run state.
- Safe interaction with local files, subprocesses, model artifacts, and optional external tools.
- Bounded resource use and explicit capability detection.
- Testable components with replaceable backends.
- Reproducible packaging and clean-environment operation.

## 3. Logical layers

### 3.1 Command Center / presentation
Owns operator-facing views, forms, validation feedback, run status, logs, metrics, artifact inventory, and action results. It should call application services rather than manipulate persistence tables or process objects directly.

### 3.2 Application services
Coordinates use cases such as preflight, workflow creation, run control, cancellation, comparison, export, diagnostics, and backup/restore. It enforces authorization and lifecycle rules at the application boundary.

### 3.3 Domain model
Defines workflow/run/stage states, configuration contracts, provenance records, capability descriptions, artifact identity, and validation rules. Domain transitions should be explicit and independently testable.

### 3.4 Workflow orchestration
Coordinates stages and dependencies, applies timeouts/retry policies, tracks progress, and ensures downstream work receives validated outputs. It should not own UI concerns or bury persistent state transitions inside long-running process code.

### 3.5 Execution adapters
Wraps training frameworks, tokenizers, data processors, inference runtimes, and native tools behind explicit interfaces. Adapters report capabilities and limitations and translate backend-specific errors into stable application-level failure categories.

### 3.6 Persistence and artifact repository
Owns durable records, manifests, run metadata, artifact references, and schema migrations. Critical writes use atomic replacement, transactions, or another documented consistency mechanism. Filesystem artifacts and database records must have explicit reconciliation rules.

### 3.7 Environment and resource services
Detects supported runtimes, drivers, hardware, disk capacity, optional tools, and resource constraints. Detection results are timestamped and distinguish observed facts from estimates and unverified capabilities.

### 3.8 Diagnostics and release evidence
Collects structured logs, status snapshots, dependency information, test reports, and package identity. Sensitive fields are redacted. Release evidence is bound to the exact candidate it describes.

## 4. Key boundaries

- Presentation code must not directly mutate workflow persistence.
- Domain rules must not depend on a specific UI framework.
- Orchestration must call execution adapters through defined interfaces.
- Execution adapters must not silently redefine lifecycle or success semantics.
- Persistence must validate schema and record versions.
- Artifact references must not be treated as proof that files exist or are valid.
- Capability detection must not be treated as proof of end-to-end support.
- Diagnostics must not expose secrets by default.
- External tools must be invoked with explicit arguments, controlled working directories, bounded timeouts, and captured exit/output state.

## 5. Run lifecycle and state model

A typical run lifecycle is:

`created → validated → queued (if queueing is supported) → running → finalizing → completed`

Alternative terminal outcomes include `failed` and `cancelled`. A `cancelling` state may be used while cleanup is in progress. State names may differ in implementation, but semantics and legal transitions must be defined.

A run may be marked `completed` only after:
1. Required stages have completed.
2. Required outputs exist and pass validation.
3. Provenance and final status have been persisted successfully.
4. Required postconditions have passed.

A process exit code is evidence, not the entire success decision.

## 6. Persistence consistency

Critical run state, shard manifests, export manifests, and other authoritative metadata must not be exposed while partially written. Suitable approaches include database transactions, write-to-temporary-file plus flush/atomic replacement, or another mechanism with equivalent guarantees on supported filesystems.

The architecture must specify:
- Which store is authoritative for each record.
- How filesystem/database disagreement is detected.
- How interrupted writes are recovered.
- How schema migrations are versioned and rolled back or recovered.
- How finalized run evidence is protected from accidental mutation.

## 7. Workflow and process control

Long-running work should be represented as managed jobs or explicit bounded executions. Each process invocation needs:
- A known executable and argument list.
- Controlled working directory and environment.
- Timeout and cancellation policy.
- Captured exit code and bounded logs.
- A defined cleanup strategy.
- A clear mapping from backend outcomes to run states.

Retries must be stage-specific. Non-idempotent work must not be automatically retried unless it has a safe resume or deduplication mechanism.

## 8. Data and artifact provenance

The architecture should link a run to its inputs, transformations, configurations, software versions, tokenizer, training outputs, evaluations, and exports. Use content hashes or immutable version identifiers where practical. Do not rely solely on mutable filenames.

Each model export is a logical package comprising the model artifact, required model-specific dataset documentation, model documentation where applicable, and a manifest that binds the parts together. Validation must detect missing or mismatched files.

## 9. Resource governance

Resource policy should be centralized enough to apply consistently across workflows. It should consider available RAM, VRAM, CPU, storage, and configured concurrency. Estimates must state uncertainty. The system should support preflight warnings and safe refusal where a workload violates hard constraints.

The architecture must not assume that GPU availability guarantees adequate VRAM, that CPU fallback will be performant, or that a developer's cached environment represents a clean installation.

## 10. Security boundaries

- Validate all user-controlled paths and prevent traversal.
- Prefer safe, documented artifact formats and deserialization paths.
- Avoid shell-string construction when an argument-list API is available.
- Redact secrets from structured logs and diagnostic exports.
- Minimize privileges and external network access.
- Treat datasets, model files, plugins, and external tool output as potentially untrusted.
- Review deletion, overwrite, executable-launch, and export operations for unintended side effects.

## 11. Configuration and compatibility

Configuration should have a versioned schema, validation errors that identify the relevant setting, explicit defaults, and migration behavior where needed. Backend compatibility must be represented as declared capability metadata rather than scattered assumptions.

## 12. Observability

Every run and stage should have stable identifiers. Logs, metrics, UI status, artifact manifests, and diagnostic bundles should share those identifiers. Logs should be structured, bounded, and redacted. User-facing status should be derived from authoritative lifecycle state rather than inferred from stale log text.

## 13. Testing strategy implications

Each layer must be testable independently. Interfaces should allow deterministic tests with fake adapters, injected process failures, controlled clocks where applicable, temporary persistence, and fault injection. End-to-end tests must still prove that the integrated product works; unit tests alone are insufficient.

## 14. Architecture decision records

Material decisions must record:
- Context and problem.
- Decision and alternatives considered.
- Consequences and tradeoffs.
- Security, reliability, resource, and compatibility implications.
- Migration or rollback approach.
- Verification needed.

Decisions affecting data integrity, persistent formats, public interfaces, export compatibility, or release packaging require explicit review.

## 15. Implementation mapping

Before treating this architecture as implemented, create a component map linking each logical responsibility to concrete modules, interfaces, persistence stores, and tests. Record deviations and rationale rather than silently weakening the baseline.
