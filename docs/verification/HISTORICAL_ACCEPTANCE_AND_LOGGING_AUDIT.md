# Historical Acceptance and Logging-Sink Audit

**Original review date:** 2026-10-11  
**Ported/curated:** 2026-10-12  
**Original review context:** branch `governance/acceptance-foundation-v1`, source snapshot `f06bbedb55e252cd7033c615733c5d2554a7d0aa`  
**Current documentation port base:** `23a3bf01975b4e4f147486cdadca51d6426684f0`  
**Evidence status:** Historical static review; not a current execution report.

## Purpose and limits

This document preserves useful findings from the 2026-10-11 static review. Its requirement-target classifications and implementation observations refer to the historical source snapshot, not automatically to current `main`. Re-check every code path and test against the candidate SHA before treating a finding as a current defect or a control as implemented.

The original reviewer could not establish a complete mapping against the authoritative v1.1.0 package and did not execute the checker or tests. No passing test result, coverage claim, or compliance conclusion is implied here.

## Acceptance-test quality findings to retain

The original review separated tests with behavioral assertions from targets that appeared structural or shallow. The key lasting rule is: **a test name or source-file-presence assertion is not evidence that a requirement's behavior is satisfied.**

When auditing acceptance coverage:

- Preserve requirement IDs and map each assertion to the exact normative statement in the authoritative catalog.
- Prefer observable behavior, including success and failure cases, over checking that a filename, substring, or import exists.
- For concurrency and lifecycle behavior, test the actual worker/process lifecycle with deterministic synchronization rather than a fake object that never executes.
- For provenance, integrity, export, and dataset-card requirements, exercise valid and invalid identities, tampering, missing fields, and mismatched artifacts.
- For release gates, prove that missing required flags and forced failures return nonzero and do not emit a success report.
- Treat a structural coverage checker as a mapping aid only. Passing it does not establish assertion quality or runtime behavior.

The historical review identified the following areas for deeper behavioral validation: Command Center authorization and response behavior; doctor/bootstrap contracts; parsed workflow and CLI configuration; dataset identity and checksums; model sizing/evaluation comparability; export/card pairing; health, retention, event and structured-logging contracts; dependency reproducibility and secret scanning; and release-gate, clean-clone, version-identity, and test-discovery behavior. These are audit prompts, not a claim that the current implementation fails them.

## Logging and secret-exposure sink inventory

The original review observed a runner path that writes child-process output to a log file and applies redaction to known values. That observation is historical and must be revalidated on the candidate SHA.

A meaningful sink audit should cover:

1. Worker stdout/stderr and file logs, including multiline and chunk-split output.
2. Event-store payloads for success, failure, cancellation, and recovery.
3. Exception messages and tracebacks from credential retrieval, subprocess startup/wait, file writes, JSON parsing, and API handlers.
4. Structured logger fields and nested dictionaries/lists.
5. API responses, health/diagnostic output, console output, and test-failure diagnostics.
6. Credential persistence, temporary/staging files, and crash-recovery state.
7. CI logs and generated test/evidence artifacts.

Use synthetic canary credentials only. Check raw and deliberately enumerated JSON, Base64, URL-encoded, multiline, and chunk-boundary forms at every applicable sink. Do not place real credentials in fixtures or assertion messages.

Known-value substitution is defense in depth, not a guarantee against arbitrary transformations or secrets the redactor never receives. Prefer allowlisted structured fields and avoid logging credential-bearing environment maps in the first place.

## Required evidence for a current conclusion

For every verification claim, record:

- Exact candidate SHA and whether the working tree was clean.
- OS, architecture, Python version, dependency environment, and relevant toolchain versions.
- Exact commands, stdout/stderr artifacts, and exit codes.
- Passed, failed, skipped, and errored test counts; preserve stack traces for failures.
- CI workflow/run URLs, job IDs, platform/Python matrix, and artifact links.
- For regression claims, equivalent commands and dependency setup on both base and candidate SHAs, with failures classified as pre-existing, regression, setup/environment, or flaky.
- For security claims, the synthetic test cases, sinks scanned, and scan result without disclosing sensitive values.

CI results, static review, structural checker output, and machine-specific native verification are distinct evidence types. Do not substitute one for another.

## Current disposition

- Preserve this audit as a historical risk and test-design reference.
- Revalidate implementation-specific observations before reporting them as current.
- Use the repository's current governance package and release evidence register as the authoritative sources for requirement IDs and status.
- Do not mark any requirement compliant based on this document alone.
