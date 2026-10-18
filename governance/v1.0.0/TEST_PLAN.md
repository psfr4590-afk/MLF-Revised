# ML-Framework Verification and Test Plan

**Document ID:** MLF-GOV-005  
**Version:** 1.0.0  
**Status:** Approved verification baseline  
**Prepared:** 2026-10-11  
**Approval:** Approved by product owner on 2026-10-11

## 1. Purpose

This plan establishes how requirements are to be verified from the beginning of development. It is a test strategy, not a report that tests have run or passed.

## 2. Verification principles

1. Every mandatory requirement has a verification method.
2. Tests verify behavior and failure semantics, not merely the existence of functions or UI controls.
3. Unit tests are necessary but do not replace integrated operator workflows.
4. The supported environment matrix is explicit.
5. Test evidence identifies the revision and relevant environment.
6. Failed, blocked, skipped, and unexecuted tests are reported distinctly.
7. Release claims never exceed the available evidence.

## 3. Test layers

### 3.1 Static and policy checks
- Formatting, linting, type checks where adopted, dependency/lockfile consistency.
- Secret scanning, dependency vulnerability checks, and source security analysis.
- Schema/configuration validation and forbidden-pattern checks for high-risk code.
- Documentation consistency and traceability validation.

### 3.2 Unit tests
- Domain state transitions and invalid transitions.
- Configuration parsing, defaults, schema validation, and migration logic.
- Path validation, redaction, manifest validation, resource admission, and comparability rules.
- Artifact and dataset-card pairing rules.
- Error mapping from execution adapters.

### 3.3 Component and integration tests
- Persistence transactions or atomic-write behavior.
- Filesystem/database reconciliation.
- Application service and orchestration behavior.
- Subprocess launch, timeout, cancellation, nonzero exit, and cleanup.
- Training/data-processing adapters with small deterministic fixtures.
- Logging correlation and diagnostic redaction.
- Export packaging and format-specific validation.

### 3.4 End-to-end tests
Exercise complete user-visible flows through the Command Center and application services, including:
- First launch and preflight.
- A non-destructive self-test.
- A minimal supported data-preparation run.
- A minimal supported training and evaluation run.
- Cancellation and recoverable failure.
- Artifact inspection and export with required documentation.
- Diagnostic bundle generation.
- Backup/restore where persistence is present.

### 3.5 Packaging and clean-environment tests
- Install and launch from a clean Git clone.
- Install and launch from a source ZIP when ZIP distribution is supported.
- Verify documented prerequisites and optional dependencies.
- Test paths containing spaces and supported non-ASCII characters where applicable.
- Confirm that no undeclared cache, global package, local executable, or developer-only file is required.
- Test uninstall/reinstall or upgrade paths where applicable.

## 4. Mandatory failure-path coverage

Tests shall cover, where relevant:
- Missing or incompatible prerequisites.
- Invalid and partially written configuration or manifests.
- Subprocess creation raising an operating-system error.
- Process timeout, crash, nonzero exit, and unexpected early exit.
- Cancellation while running and during finalization.
- Database/file write failure and interrupted atomic replacement.
- Disk full or insufficient free space.
- Missing, corrupt, truncated, or mismatched artifacts.
- Missing or mismatched dataset documentation.
- Invalid schema version or migration failure.
- Secret-bearing values in logs and diagnostic bundles.
- Path traversal attempts and destructive-action boundary violations.
- Backend unavailable, capability mismatch, and GPU-to-CPU fallback where supported.
- Incomplete output with a misleadingly successful process exit.
- Retry of idempotent and non-idempotent stages.

Each test must assert the resulting state, retained evidence, and whether recovery is safe.

## 5. Environment and resource matrix

The project owner must approve a support matrix before release claims are made. The matrix should cover relevant dimensions, such as:
- Supported operating systems and runtime versions.
- CPU-only versus GPU-enabled execution where claimed.
- Supported accelerator/backend versions.
- Minimum and recommended RAM/VRAM/storage.
- Supported model/configuration sizes.
- Required and optional external tools.
- Clean clone versus source archive.
- Offline versus network-required operations.

Do not multiply every dimension blindly. Select combinations based on risk, support promises, and known incompatibilities. Mandatory combinations must have explicit evidence; untested combinations remain unverified.

## 6. Resource and endurance tests

- Validate that preflight reports constraints and uncertainty accurately enough for its stated purpose.
- Test bounded concurrency and safe refusal or warning when configured limits are exceeded.
- Test long-running log growth, temporary-file cleanup, and storage exhaustion handling.
- Exercise representative small, medium, and maximum-supported workloads.
- Record peak RAM/VRAM, storage use, duration, and relevant environment details.
- Test interruption and recovery without claiming that all checkpoints or partial outputs are resumable.

## 7. ML correctness and evidence

For each supported training/evaluation path:
- Verify dataset and tokenizer identity.
- Verify configuration and backend versions are recorded.
- Verify the intended stages ran and produced validated outputs.
- Check loss/metric values for validity, not just presence.
- Use a deterministic fixture or known baseline where feasible.
- Document nondeterminism, numerical tolerance, and hardware-dependent variation.
- Keep performance benchmarks separate from functional acceptance.
- Do not treat a short smoke run as evidence of convergence or model quality.

## 8. Security and privacy verification

- Test path traversal and invalid path handling.
- Inspect subprocess argument construction and environment handling.
- Verify redaction using seeded test secrets.
- Check that diagnostic bundles include only disclosed and necessary information.
- Verify no hidden network transfer in local workflows.
- Review dependency and packaged executable provenance.
- Test unsafe or unsupported artifact inputs fail without uncontrolled execution.

## 9. Test evidence format

Each run of the test suite should record:
- Test-plan version and requirement IDs covered.
- Source revision and working-tree status.
- OS, runtime, dependency, backend, and hardware details relevant to the test.
- Commands or test entry points used.
- Passed, failed, blocked, skipped, and not-run counts.
- Failure logs and artifact references.
- Known limitations and deviations.
- Package identity when testing a release candidate.

Evidence must be retained and reproducible to the extent practical. Redact secrets and private user data.

## 10. Release verification sequence

1. Validate the source tree and dependency metadata.
2. Run static/security checks and unit tests.
3. Run component/integration tests and failure-path tests.
4. Run Command Center end-to-end workflows.
5. Build the release artifact from the declared revision.
6. Install the artifact in a clean supported environment.
7. Exercise mandatory workflows on the approved matrix.
8. Validate exported artifacts and paired documentation.
9. Generate a release evidence report tied to the exact package.
10. Apply the release gate in `ACCEPTANCE_CRITERIA.md`.

The sequence may be automated, but automation does not waive evidence requirements.

## 11. Defect severity and disposition

Defects should be classified by impact on security, data integrity, core workflow, compatibility, usability, and release claims. A severity policy must be approved by the project owner. Mandatory failures cannot be waived by relabeling them as low priority; any waiver requires documented risk acceptance and must not contradict a hard release criterion.

## 12. Completion rules

A requirement is verified only when its defined test or inspection has passed against the intended implementation and relevant environment. A test that was not run is not a pass. A passing unit test does not establish end-to-end support. A smoke test does not establish convergence, performance, or general model quality.
