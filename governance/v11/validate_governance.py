#!/usr/bin/env python3
"""Structural and checksum validator for ML-Framework governance v1.1.0 catalog."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

V11_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "requirements_catalog.csv",
    "traceability_matrix.csv",
    "test_catalog.csv",
    "SHA256SUMS-HISTORY.txt",
    "SHA256SUMS.txt",
    "validate_governance.py",
    "check_test_coverage.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_checksum_file(checksum_file: Path) -> list[str]:
    errors: list[str] = []
    for raw in checksum_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            errors.append(f"Malformed checksum line in {checksum_file.name}: {line}")
            continue
        expected, rel = parts[0], parts[1].lstrip("* ")
        target = V11_DIR / rel
        if not target.is_file():
            errors.append(f"Missing checksummed file: {rel}")
        elif _sha256(target) != expected:
            errors.append(f"Checksum mismatch for {rel}: expected {expected}, got {_sha256(target)}")
    return errors


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (V11_DIR / name).is_file():
            errors.append(f"Missing required governance file: {name}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    errors.extend(_verify_checksum_file(V11_DIR / "SHA256SUMS-HISTORY.txt"))
    errors.extend(_verify_checksum_file(V11_DIR / "SHA256SUMS.txt"))

    with (V11_DIR / "requirements_catalog.csv").open(encoding="utf-8") as f:
        reqs = list(csv.DictReader(f))
    with (V11_DIR / "traceability_matrix.csv").open(encoding="utf-8") as f:
        trace = list(csv.DictReader(f))
    with (V11_DIR / "test_catalog.csv").open(encoding="utf-8") as f:
        tests = list(csv.DictReader(f))

    req_ids = [r["requirement_id"] for r in reqs]
    trace_req_ids = [r["requirement_id"] for r in trace]
    test_ids = [r["test_id"] for r in tests]

    if len(set(req_ids)) != 64 or len(reqs) != 64:
        errors.append(f"Expected 64 unique requirements, found {len(set(req_ids))}")
    if len(trace) != 64 or set(trace_req_ids) != set(req_ids):
        errors.append("Traceability matrix does not map all 64 requirements 1-to-1")
    if len(set(test_ids)) != 64 or len(tests) != 64:
        errors.append(f"Expected 64 unique test IDs, found {len(set(test_ids))}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    print(
        f"Governance v1.1.0 validation PASS: {len(reqs)} requirements, "
        f"{len(trace)} traceability rows, {len(tests)} unique test IDs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
