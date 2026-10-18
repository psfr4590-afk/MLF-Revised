# ML-Framework Product Charter

**Document ID:** MLF-GOV-001  
**Version:** 1.0.0  
**Status:** Approved project baseline  
**Prepared:** 2026-10-11  
**Approval:** Approved by product owner on 2026-10-11

## 1. Purpose

This charter establishes why ML-Framework exists, whom it serves, the outcomes it is intended to deliver, and the principles that govern product decisions. It is a foundational design document, informed by lessons learned during development. It is not a report of current implementation status.

## 2. Product vision

ML-Framework is a local-first machine-learning workbench that helps an operator prepare data, configure and execute model workflows, observe progress, evaluate results, and manage model artifacts through one coherent browser-based Command Center.

The product should make complex ML workflows understandable, repeatable, inspectable, and manageable on ordinary supported hardware without pretending that every model or workload fits every machine.

## 3. Intended users

- Individual developers and operators running ML workflows on their own computers.
- Users who need understandable control over data preparation, training, evaluation, and model artifact handling.
- Technical users who value inspectable behavior, reproducible evidence, and explicit limits over opaque automation.
- Maintainers responsible for packaging, troubleshooting, and releasing the application.

## 4. Product outcomes

The product is successful when an operator can, within a declared support envelope:

1. Install and launch it from documented prerequisites on a clean supported environment.
2. Understand available capabilities, prerequisites, hardware constraints, and unsupported combinations before committing substantial resources.
3. Configure and run supported workflows through one primary browser Command Center.
4. Follow a run from inputs through stages, outputs, metrics, and logs using stable identifiers and provenance.
5. Identify failures, recover safely where supported, and preserve useful evidence for diagnosis.
6. Evaluate results with clearly stated metrics, limitations, and comparability rules.
7. Export supported artifacts with the metadata and documentation required to understand and use them.
8. Verify a release from repeatable evidence tied to an exact source revision and package.

## 5. Product principles

### 5.1 Local-first and operator-controlled
Core workflows should run locally unless a feature explicitly documents an external dependency. External services, uploads, telemetry, and network access must be disclosed and governed by explicit configuration and consent.

### 5.2 One primary operator interface
The browser Command Center is the canonical operator interface. Other entry points, such as scripts or command-line utilities, may support development, automation, or recovery, but must not become competing primary user experiences.

### 5.3 Evidence before claims
The application must distinguish supported, detected, configured, running, completed, and verified states. Documentation and UI must not imply that a capability works merely because code or a control exists.

### 5.4 Safe failure and recoverable state
Failures must not silently corrupt durable state or misrepresent incomplete work as successful. Writes to critical manifests and state transitions should be atomic or use an equivalently justified consistency strategy.

### 5.5 Resource honesty
Hardware detection and estimates must be treated as estimates, not guarantees. Workflows should expose relevant CPU, RAM, VRAM, storage, power, thermal, and time constraints where measurable and useful.

### 5.6 Traceability and reproducibility
Inputs, configuration, software versions, random seeds where applicable, dataset identity, tokenizer identity, artifacts, and results should be recorded sufficiently to understand how a result was produced. Reproducibility claims must state their scope and limitations.

### 5.7 Security by design
Use least privilege, safe path handling, secret redaction, dependency hygiene, safe deserialization practices, and explicit boundaries around local execution. Diagnostic output must not casually expose private data or credentials.

### 5.8 Accessible explanations
Errors should explain what failed, the likely impact, and a safe next action. Technical detail may be available without making basic operation depend on reading source code.

### 5.9 Small, testable increments
Requirements and architecture should support vertical slices that can be verified end to end before additional complexity is added.

## 6. Product boundaries

The product focuses on local machine-learning workflow orchestration and operator visibility. It does not promise that arbitrary models can train successfully on arbitrary hardware, that training results are useful without evaluation, or that exported models are safe for every application.

Cloud-scale training, unattended autonomous modification of the host, undisclosed telemetry, and unbounded resource consumption are outside the default product contract.

## 7. Success measures

Measures should be defined and baselined during planning, then tracked with evidence. Candidate measures include:
- Clean-environment installation and launch success across the declared support matrix.
- Percentage of mandatory operator workflows passing end-to-end tests.
- Rate of state or manifest corruption under injected failures.
- Percentage of runs with complete provenance for required fields.
- Recovery success for documented recoverable failure modes.
- Resource-estimate accuracy and frequency of unsafe oversubscription.
- Release evidence completeness and reproducibility.

No numerical target is considered approved until explicitly set by the project owner.

## 8. Decision authority and change control

The project owner approves this charter, product scope, mandatory requirements, support claims, and release policy. Architecture decisions that materially affect security, data integrity, compatibility, or user workflows must be documented and reviewed before implementation. Changes to approved requirements must record rationale, affected components, verification impact, and migration needs.

## 9. Relationship to other governance documents

- `SCOPE.md` defines what is in and out.
- `REQUIREMENTS.md` defines observable system obligations.
- `ARCHITECTURE.md` defines intended structure and technical boundaries.
- `TEST_PLAN.md` defines verification strategy.
- `TRACEABILITY.md` connects requirements, design, implementation, and evidence.
- `ACCEPTANCE_CRITERIA.md` defines the conditions for accepting work and releases.

## 10. Status and effective date

This version 1.0.0 baseline was approved by the project owner on 2026-10-11 and is effective as the governing product charter from that date. Approval establishes intended product direction; it does not assert that the current implementation satisfies the charter.
