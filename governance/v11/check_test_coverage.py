#!/usr/bin/env python3
"""Verify that every requirement test_id in traceability_matrix.csv exists as a real test function."""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import sys

V11_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check governance requirement test coverage")
    parser.add_argument("--repo-root", type=Path, default=V11_DIR.parents[1])
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()

    with (V11_DIR / "traceability_matrix.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing: list[str] = []
    parsed_cache: dict[Path, set[str]] = {}
    for row in rows:
        req_id = row["requirement_id"]
        test_id = row["test_id"]
        test_file = root / row["test_file"]
        if not test_file.is_file():
            missing.append(f"{req_id}: missing file {row['test_file']}")
            continue
        if test_file not in parsed_cache:
            tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
            parsed_cache[test_file] = {
                node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            }
        if test_id not in parsed_cache[test_file]:
            missing.append(f"{req_id}: missing function {test_id} in {row['test_file']}")

    if missing:
        for item in missing:
            print(f"MISSING: {item}", file=sys.stderr)
        return 2

    print(f"Governance test coverage PASS: all {len(rows)} requirement test targets exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
