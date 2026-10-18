# ML-Framework Requirements Traceability

**Document ID:** MLF-GOV-006  
**Version:** 1.0.0  
**Status:** Approved project baseline  
**Prepared:** 2026-10-11  
**Approval:** Approved by product owner on 2026-10-11

## 1. Purpose

Traceability links the product goals to requirements, design responsibilities, implementation, tests, and evidence. It is intended to be established before implementation grows, so features cannot become accepted merely because they exist.

This document defines the traceability method and initial requirement map. It does not assert that all referenced implementation components or tests already exist.

## 2. Required chain

Every mandatory capability should be traceable through:

`Product outcome → Requirement ID → Architecture responsibility/decision → Implementation location → Verification case → Evidence → Acceptance decision`

Missing links must be visible. A missing implementation link means the requirement is not mapped. A missing test link means it is not adequately verifiable. Missing evidence means it is not verified.

## 3. Record fields

Maintain a traceability register with these fields:
- Requirement ID and short name.
- Priority and approval status.
- Source/rationale.
- Architecture component or ADR.
- Implementation module/interface.
- Test case or verification method.
- Evidence reference and source revision.
- Status and known gaps.
- Owner and last-updated date.

## 4. Status vocabulary

Use explicit statuses:
- `PROPOSED`: under consideration.
- `APPROVED`: accepted baseline requirement.
- `MAPPED`: implementation and verification locations identified.
- `IMPLEMENTED`: implementation is present, not necessarily proven.
- `VERIFIED`: required verification passed with evidence.
- `BLOCKED`: verification or implementation cannot proceed.
- `DEFERRED`: intentionally postponed with approval.
- `REJECTED`: considered and not accepted.

Do not collapse `IMPLEMENTED` and `VERIFIED`.

## 5. Initial traceability map

The following is the intended mapping at the responsibility level. Concrete file paths, test names, and evidence IDs must be added as implementation decisions are made.

| Requirement family | Requirement IDs | Architecture responsibility | Minimum verification |
|---|---|---|---|
| Primary interface | REQ-UI-001–006 | Command Center, application services | UI end-to-end workflow and action/state tests |
| Environment/setup | REQ-ENV-001–006 | Environment services, packaging | Clean clone and ZIP install tests, preflight tests |
| Workflow lifecycle | REQ-WF-001–008 | Domain model, orchestrator, adapters | State transition, subprocess, cancellation, retry tests |
| Data integrity | REQ-DATA-001–008 | Persistence, provenance, artifact repository | Fault injection, schema/migration, lineage tests |
| ML/resource behavior | REQ-ML-001–008 | Resource services, training/evaluation adapters | Resource matrix, fixture runs, metrics/provenance checks |
| Artifacts/exports | REQ-ART-001–007 | Artifact repository, export services | Format validation and paired-documentation integrity |
| Operations | REQ-OPS-001–008 | Diagnostics, persistence, cleanup services | Redaction, recovery, backup/restore, cleanup preview tests |
| Security/privacy | REQ-SEC-001–006 | Security boundary, adapters, packaging | Threat-driven tests, secret scans, path tests, network review |
| Release verification | REQ-REL-001–007 | Build/package pipeline, test evidence | Exact-revision clean-install and release-gate evidence |

## 6. Product-outcome traceability

| Product outcome | Primary requirements |
|---|---|
| Install and launch cleanly | REQ-ENV-001–006, REQ-REL-004 |
| Understand capability and limits | REQ-UI-002, REQ-UI-005, REQ-ML-001–003 |
| Operate supported workflows in one interface | REQ-UI-001–004, REQ-WF-001–008 |
| Trace a run from input to artifact | REQ-WF-002, REQ-DATA-001–006, REQ-ART-001 |
| Diagnose and recover safely | REQ-OPS-001–008, REQ-WF-004–007 |
| Interpret and compare results | REQ-ML-004–007 |
| Export a documented model | REQ-ART-001–007 |
| Verify releases with evidence | REQ-REL-001–007 |

## 7. Change impact analysis

When a requirement changes:
1. Record the rationale and approval.
2. Identify affected architecture responsibilities and ADRs.
3. Identify implementation and persistent-schema impacts.
4. Identify tests that must be added or updated.
5. Identify documentation and support-claim changes.
6. Re-run affected verification and record evidence.
7. Update the traceability register.

Changes to serialization, manifests, run lifecycle, artifact formats, or persistent schemas require explicit migration and backward-compatibility analysis.

## 8. Orphan detection

Regularly identify:
- Approved requirements with no implementation mapping.
- Implemented features with no approved requirement or documented rationale.
- Requirements with no verification method.
- Passing tests that do not map to a requirement or risk.
- Release claims without supporting evidence.
- Artifacts without provenance or required documentation.
- Architecture decisions with no identified verification impact.

Orphans are reviewed; they are not automatically deleted or accepted.

## 9. Release traceability rule

A release report must identify the exact candidate revision and package and summarize mandatory requirement evidence using the controlled outcomes `PASS`, `FAIL`, `BLOCKED`, `SKIPPED`, or `NOT RUN`. Requirement lifecycle statuses such as `DEFERRED` describe planning disposition and must not be used as test-evidence outcomes. Aggregate test counts do not substitute for requirement-level evidence.

## 10. Maintenance

The traceability register should be updated as part of feature development and code review, not postponed until release preparation. The project owner defines the storage location and required review cadence.
