"""Import retained JSON run manifests into the Phase 6 SQLite experiment store.

The importer is intentionally idempotent: it never deletes raw manifests or
other evidence and re-projects each retained run into the normalized database.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.experiment_db import ExperimentDB


def import_runs(output_dir: Path, *, run_id: str | None = None) -> int:
    output_dir = Path(output_dir).resolve()
    run_root = output_dir / "runs"
    manifests = []
    if run_id:
        candidate = run_root / run_id / "run_manifest.json"
        if candidate.is_file():
            manifests = [candidate]
    elif run_root.is_dir():
        manifests = sorted(run_root.glob("*/run_manifest.json"))

    db = ExperimentDB(output_dir / "experiment.db")
    imported = 0
    try:
        for path in manifests:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid run manifest: {path}") from exc
            if not isinstance(manifest, dict) or not manifest.get("run_id"):
                raise RuntimeError(f"Run manifest missing run_id: {path}")
            db.sync_manifest(manifest)
            imported += 1
    finally:
        db.close()
    return imported


def main() -> int:
    parser = argparse.ArgumentParser(description="Import retained Model Lab run manifests into experiment.db")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    count = import_runs(Path(args.output_dir), run_id=args.run_id)
    print(f"Imported {count} run(s) into {Path(args.output_dir).resolve() / 'experiment.db'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
