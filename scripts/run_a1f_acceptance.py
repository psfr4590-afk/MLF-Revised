#!/usr/bin/env python3
"""Run the local A1-F/A2/A4/A5 acceptance suite and emit machine-readable status."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(ROOT)
    full = os.getenv("MLFRAMEWORK_RUN_FULL_A1F") == "1"
    commands = [[
        sys.executable, "-m", "pytest", "-q",
        "tests/acceptance/test_a1_filesystem_end_to_end.py",
        "tests/acceptance/test_a2_provenance.py",
        "tests/acceptance/test_a4_resume.py",
        "tests/acceptance/test_a5_export_validation.py",
        "tests/test_semantic_dedup_offline.py",
        "tests/test_shard_integrity.py",
        "tests/test_training_provenance_lineage.py",
    ]]
    report = {
        "schema": 1,
        "suite": "ML-Framework local acceptance",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "github_modified": False,
        "network_policy": "network calls blocked in the crawler/full-pipeline tests; sources explicitly local-only",
        "full_pipeline_requested": full,
        "commands": [],
    }
    overall = 0
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report["commands"].append({"argv": command, "returncode": result.returncode, "output": result.stdout})
        print(result.stdout, end="")
        if result.returncode:
            overall = result.returncode
            break
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = ("PASS_TEST_COMMANDS" if overall == 0 else "FAIL")
    report["acceptance_certification"] = "NOT_CERTIFIED_BY_FAST_GATE" if not full and overall == 0 else ("REVIEW_FULL_EVIDENCE_REQUIRED" if overall == 0 else "FAIL")
    report["exit_code"] = overall
    report_path = ROOT / ".acceptance" / "acceptance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nMachine-readable report: {report_path}")
    print(f"Test command status: {report['status']}")
    print(f"Acceptance certification: {report['acceptance_certification']}")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
