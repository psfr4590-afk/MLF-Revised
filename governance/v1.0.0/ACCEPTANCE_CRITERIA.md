# ML-Framework Acceptance Criteria

**Document ID:** MLF-GOV-007  
**Version:** 1.0.0  
**Status:** Approved project baseline  
**Prepared:** 2026-10-11  
**Approval:** Approved by product owner on 2026-10-11

## 1. Purpose

This document defines how work is accepted. It separates implementation completion from verified behavior and separates feature acceptance from release certification. These criteria are intended to govern development from the beginning; they do not certify the current repository.

## 2. General acceptance rules

A deliverable is acceptable only when:
- Its approved requirements are identified.
- Observable behavior matches those requirements.
- Relevant positive, negative, and boundary cases are tested.
- Errors and failure states are understandable and safe.
- Security, privacy, resource, and data-integrity implications are addressed.
- Documentation is updated.
- Evidence is tied to the tested revision and environment.
- Known limitations are recorded rather than concealed.

Code review approval, a successful build, or a passing unit-test count alone is not sufficient evidence of complete acceptance.

## 3. Feature-level acceptance

For each feature, define before implementation:
1. User problem and intended outcome.
2. Preconditions and supported environment.
3. Normal behavior and observable output.
4. Invalid input and failure behavior.
5. Cancellation, retry, and recovery behavior where relevant.
6. Data persistence and provenance impact.
7. Security/privacy impact.
8. Resource and performance constraints.
9. Verification cases and required evidence.
10. Documentation changes.

Feature acceptance requires all mandatory criteria to pass. A criterion that does not apply must be marked not applicable with a rationale. A criterion that has not been tested remains unverified.

## 4. Core product acceptance

### 4.1 Installation and startup
- Documented prerequisites and support envelope are available.
- Installation and launch succeed in a clean supported environment.
- ZIP-based source installation is tested if offered as supported.
- Preflight identifies missing required dependencies and distinguishes optional capabilities.
- First launch does not depend on undeclared local files or developer-specific cache.

### 4.2 Command Center
- The browser Command Center is the one primary operator interface.
- Core actions provide visible outcomes and meaningful errors.
- Status shown in the UI matches authoritative run state.
- Logs, metrics, stage state, and artifacts correlate to the correct run.
- Unavailable or unverified features are not presented as proven capabilities.

### 4.3 Workflow integrity
- Supported workflows validate configuration and prerequisites before execution.
- Every run has stable identity and traceable inputs/outputs.
- Invalid state transitions are rejected or explicitly handled.
- Cancellation, timeout, subprocess failure, and nonzero exit produce accurate states.
- Incomplete output cannot be reported as a successful run.
- Critical manifests and state survive injected write failures without partial records being mistaken for valid records.

### 4.4 ML run evidence
- The run records required configuration, input identity, backend/software versions, and metrics.
- Metrics are validated and have documented interpretation.
- Resource guidance states uncertainty and does not promise success.
- Model/configuration support claims are tied to the approved matrix.
- A smoke test is not represented as proof of convergence or model quality.

### 4.5 Artifact and export integrity
- Supported exports pass format-specific validation.
- Model artifact and required model-specific dataset documentation are present as one logical export unit.
- A manifest or equivalent integrity check detects missing or mismatched components.
- Provenance identifies the run and relevant inputs/configuration.
- Compatibility claims are supported by tests against the named runtime or format.

### 4.6 Security and privacy
- Path traversal and unintended destructive operations are prevented.
- Secrets are absent from normal logs and redacted diagnostic bundles.
- External transfers and services are disclosed.
- Packaged dependencies and executables have documented provenance appropriate to the release.
- Unsafe or unsupported artifact inputs fail safely.

### 4.7 Operations and recovery
- Known failures produce useful diagnostic context.
- Diagnostic bundles disclose what they contain and redact sensitive values.
- Backup/restore works for persistent data that the product promises to preserve.
- Cleanup previews its impact and respects retention rules.
- Recovery procedures state any data or progress that cannot be recovered.

## 5. Release acceptance

A release candidate is eligible for certification only when all of the following are true:
- The candidate revision and package identity are fixed and recorded.
- Mandatory static, unit, integration, end-to-end, security, and packaging checks pass as defined by the approved release plan.
- Mandatory supported-environment combinations have evidence.
- Clean-environment installation and launch pass.
- Core operator workflows pass end to end.
- Artifact exports and required documentation pass integrity checks.
- No unresolved critical or high-severity defect remains under the approved severity policy, unless an explicitly permitted, documented risk acceptance exists and does not violate a hard gate.
- Traceability is complete for all mandatory requirements.
- Test and release evidence is retained and reviewable.
- Release notes accurately describe supported capabilities, limitations, and known issues.

## 6. Hard NO-GO conditions

Release certification must be withheld if any of these conditions applies:
- A mandatory criterion is failed, blocked, or not tested.
- The tested package cannot be tied to the declared source revision.
- Clean-environment installation or a mandatory core workflow fails.
- Provenance or artifact/documentation pairing is missing, corrupt, or mismatched.
- Security or data-integrity behavior violates an approved mandatory requirement.
- The release claims functionality, compatibility, convergence, or performance beyond the evidence.
- Required evidence is missing or cannot be reviewed.

A release owner cannot turn a hard NO-GO into a pass by changing labels or citing a high aggregate test count.

## 7. Waivers and exceptions

A waiver must identify the criterion, reason, affected users, risk, mitigation, expiry/review date, and approving authority. Hard NO-GO conditions are not waivable unless the governing release policy explicitly permits the exception and the exception does not misrepresent the release as fully certified.

## 8. Evidence requirements

Evidence should include, as relevant:
- Source revision and working-tree state.
- Build/package hash and build inputs.
- Environment and dependency versions.
- Test commands and results.
- Logs for failures and recovery behavior.
- Export manifests and validation results.
- Traceability status and known limitations.

Evidence must distinguish passed, failed, blocked, skipped, and not run.

## 9. Acceptance decision record

For each milestone or release, record:
- Candidate name/version and exact revision.
- Criteria and requirements covered.
- Pass/fail/blocked/not-tested summary.
- Defects, deviations, and approved waivers.
- Evidence locations.
- Decision, approver, and date.

## 10. Effective use

These criteria are the governing acceptance baseline approved by the project owner on 2026-10-11. Approval does not imply that any existing feature has passed; each implementation must be evaluated against the criteria independently.
