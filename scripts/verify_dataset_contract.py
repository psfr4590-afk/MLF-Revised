"""Static verification for the ten production dataset definitions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.dataset_contract import validate_dataset_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all Model Lab dataset contracts without crawling")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    try:
        report = validate_dataset_contract(
            ROOT,
            ROOT / "config" / "dataset_groups.yaml",
            ROOT / "config" / "dataset_profiles.yaml",
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"DATASET CONTRACT FAILED: {exc}")
        return 1

    report["status"] = "passed"
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("DATASET CONTRACT PASSED")
        print(f"groups={report['groups']} profiles={report['profiles']}")
        print(f"source_kinds={','.join(report['source_kinds'])}")
        print(f"executable_exclusion_tags={len(report['executable_exclusion_tags'])}")
        print(f"production_revision_ready={report['production_revision_ready']}")
        if report["mutable_revisions"]:
            print("mutable_revisions:")
            for item in report["mutable_revisions"]:
                print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
