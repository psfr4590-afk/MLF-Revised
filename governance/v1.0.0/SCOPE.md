# ML-Framework Scope Definition

**Document ID:** MLF-GOV-002  
**Version:** 1.0.0  
**Status:** Approved project baseline  
**Prepared:** 2026-10-11  
**Approval:** Approved by product owner on 2026-10-11

## 1. Purpose

This document establishes the intended product boundary and prioritization rules. It is a forward-looking scope baseline informed by development experience, not an inventory of features currently implemented.

## 2. In scope

### 2.1 Core operator experience
- One browser-based Command Center as the primary operator interface.
- Clear application health, capability, prerequisite, and support information.
- Configuration and execution of supported workflows.
- Run status, stage progress, logs, metrics, errors, and artifact visibility.
- Clear distinction between actions that are available, unavailable, running, complete, failed, or not verified.

### 2.2 Machine-learning workflow
- Data intake and preparation workflows supported by declared connectors or local inputs.
- Cleaning, deduplication, weighting, tokenization, and sharding where enabled by the selected pipeline.
- Training orchestration for explicitly supported model/configuration combinations.
- Evaluation and comparison of runs using named metrics and documented comparability rules.
- Export and validation of supported model formats and associated metadata.
- Inference or integration workflows only where explicitly supported and tested.

### 2.3 Reliability and operations
- Preflight checks and non-destructive self-tests.
- Stable run and stage identifiers, structured logs, and correlated diagnostic evidence.
- Atomic persistence or equivalent consistency guarantees for critical state and manifests.
- Explicit cancellation, timeout, retry, and failure-state semantics.
- Backup, restore, and versioned schema migration for persistent user data where persistence is required.
- Safe cleanup with preview, retention rules, and storage-impact visibility.
- Redacted diagnostic bundles and understandable health/readiness information.
- Reproducible release verification tied to an exact revision and distributable package.

### 2.4 Security and privacy
- Local-first default behavior and clear disclosure of network/external-service use.
- Secret and sensitive-path redaction in logs and diagnostics.
- Safe path validation and restricted file operations.
- Safe model/checkpoint loading practices appropriate to supported formats.
- Dependency and packaging integrity checks.
- No hidden telemetry or undisclosed upload of user data.

### 2.5 Documentation and onboarding
- Clean-machine installation instructions with supported prerequisites.
- Honest hardware and compatibility guidance.
- Beginner-readable explanation of the workflow and major components.
- Troubleshooting guidance mapped to known error categories.
- Model and dataset documentation that follows exported artifacts.

## 3. Out of scope by default

The following are not guaranteed by the core product contract unless separately approved:
- Cloud-hosted or distributed training.
- Guaranteed training of arbitrary model sizes on arbitrary hardware.
- Automatic acquisition of paid services or cloud resources.
- Unattended modification of operating-system settings, drivers, firmware, or security controls.
- Unbounded background job execution.
- Silent upload of datasets, checkpoints, logs, or telemetry.
- Claims that generated model outputs are accurate, unbiased, safe, or suitable for a high-stakes purpose without task-specific evidence.
- Multiple competing primary operator interfaces.

## 4. Prioritization

### P0: Product integrity
Data integrity, security, clear failure semantics, operator control, provenance, and a working primary interface.

### P1: End-to-end core workflows
Clean installation, preflight, data preparation, supported training, evaluation, artifact management, export validation, and recovery paths.

### P2: Operational quality
Run comparison, diagnostic bundles, resource-aware guidance, backup/restore, retention management, and richer capability reporting.

### P3: Optional extensions
Additional backends, formats, advanced scheduling, remote integrations, and convenience features. Each extension requires explicit support claims and its own acceptance evidence.

Priority labels describe planning order, not the current implementation state.

## 5. Scope entry and exit rules

A proposed feature enters scope only when it has:
1. A user or operational problem statement.
2. A defined owner and priority.
3. Requirements and failure behavior.
4. Security, privacy, data-integrity, and resource-impact review appropriate to the feature.
5. A verification approach and acceptance criteria.
6. Documented dependencies and support limitations.

A feature is not complete merely because it is implemented. Completion requires evidence against its approved acceptance criteria.

## 6. Scope change control

Scope changes must identify the reason, user impact, dependencies, risks, schedule/resource implications, affected requirements, and verification updates. Changes that expand network access, destructive behavior, data handling, model formats, or execution privileges require explicit review.

## 7. Deferred does not mean promised

Items in P2 or P3 are candidates, not commitments. They become committed scope only after approval and planning. This prevents a list of possible features from quietly becoming a list of obligations.
