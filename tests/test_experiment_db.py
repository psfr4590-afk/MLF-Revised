from __future__ import annotations

import json
import sqlite3

from pipeline.experiment_db import ExperimentDB


def _manifest() -> dict:
    return {
        "schema": 1,
        "run_id": "run_test_001",
        "experiment_id": "exp:test",
        "dataset_id": "dataset_test",
        "timestamps": {"started": "2026-10-06T10:00:00+00:00", "updated": "2026-10-06T10:01:00+00:00", "ended": "2026-10-06T10:02:00+00:00"},
        "git": {"commit": "abc123", "branch": "test", "dirty": False},
        "hardware": {"host": {"platform": "test", "cpu_name": "CPU", "cpu_threads": 12, "total_ram_gb": 8.0}, "torch": {"version": "2.0", "device_count": 0}},
        "configuration": {"train_config_sha256": "cfg123"},
        "dataset_identity": {"dataset_id": "dataset_test", "requested_group": "ai_ml"},
        "stages": {
            "crawl": {
                "status": "PASS", "start": "2026-10-06T10:00:00+00:00", "end": "2026-10-06T10:00:10+00:00",
                "duration_seconds": 10.0, "metrics": {"document_count": 4, "token_count": 100},
            }
        },
        "artifacts": [{"stage": "crawl", "path": "scratch/01_crawled.jsonl", "sha256": "artifact123"}],
        "warnings": [],
        "errors": [],
        "final_status": "PASS",
    }


def test_database_persists_and_reopens(tmp_path):
    path = tmp_path / "output" / "experiment.db"
    db = ExperimentDB(path)
    db.sync_manifest(_manifest())
    db.close()

    reopened = ExperimentDB(path)
    run = reopened.get_run("run_test_001")
    assert run is not None
    assert run["status"] == "PASS"
    assert reopened.search_runs(status="PASS")[0]["id"] == "run_test_001"
    lineage = reopened.lineage("run_test_001")
    assert lineage["dataset"][0]["id"] == "dataset_test"
    assert lineage["stages"][0]["stage_name"] == "crawl"
    assert lineage["artifacts"][0]["sha256"] == "artifact123"
    reopened.close()


def test_schema_retains_raw_evidence_relationships(tmp_path):
    path = tmp_path / "experiment.db"
    db = ExperimentDB(path)
    db.sync_manifest(_manifest())
    tables = {
        row[0]
        for row in sqlite3.connect(path).execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"experiments", "runs", "datasets", "sources", "stages", "artifacts", "checkpoints", "metrics", "evaluations",
            "hardware", "configs", "warnings", "errors", "runtime_estimates"} <= tables
    assert json.loads(db.conn.execute("SELECT snapshot_json FROM hardware").fetchone()[0])["host"]["platform"] == "test"
    db.close()


def test_compare_runs_returns_structured_evidence(tmp_path):
    db = ExperimentDB(tmp_path / "experiment.db")
    first = _manifest()
    second = _manifest()
    second["run_id"] = "run_test_002"
    second["git"]["commit"] = "def456"
    db.sync_manifest(first)
    db.sync_manifest(second)
    comparison = db.compare_runs("run_test_001", "run_test_002")
    assert {row["id"] for row in comparison["runs"]} == {"run_test_001", "run_test_002"}
    db.close()


def test_external_evidence_is_projected_into_sqlite(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    db = ExperimentDB(output / "experiment.db")
    manifest = _manifest()
    manifest["metrics"] = {}
    manifest["metrics"]["train"] = {
        "val_loss": 1.25,
        "step": 20,
        "training_duration_seconds": 120.0,
        "initial_estimate_seconds": 150.0,
        "tokens_per_sec": 42.0,
        "steps_per_sec": 0.5,
    }
    manifest["stages"]["train"] = {
        "status": "PASS",
        "start": "2026-10-06T10:00:00+00:00",
        "end": "2026-10-06T10:02:00+00:00",
        "duration_seconds": 120.0,
        "metrics": manifest["metrics"]["train"],
    }
    db.sync_manifest(manifest)

    (output / "logs").mkdir()
    (output / "logs" / "metrics.jsonl").write_text(
        json.dumps({"step": 10, "train_loss": 2.0, "tokens_per_sec": 40.0}) + "\n"
        + json.dumps({"step": 20, "train_loss": 1.25, "tokens_per_sec": 42.0}) + "\n",
        encoding="utf-8",
    )
    (output / "checkpoints").mkdir()
    checkpoint = output / "checkpoints" / "ckpt_final.pt"
    checkpoint.write_bytes(b"checkpoint")
    (output / "checkpoints" / "ckpt_final.pt.manifest.json").write_text(
        json.dumps({
            "schema": 3,
            "kind": "checkpoint",
            "checkpoint_kind": "final",
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
            "step": 20,
            "val_loss": 1.25,
            "best_val_loss": 1.25,
            "best_step": 20,
            "final_step": 20,
        }),
        encoding="utf-8",
    )
    (output / "preflight_report.json").write_text(
        json.dumps({
            "estimate": {"estimated_duration_seconds": 150.0},
            "benchmark": {"tokens_per_sec": 40.0},
        }),
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps({
            "retrieval": [{
                "status": "SUCCESS",
                "source": {"kind": "web", "identifier": "https://example.test", "revision": None,
                           "license": "unknown", "raw_source_sha256": None},
                "stats": {"requests": 2, "successes": 2, "failures": 0, "documents": 4,
                          "retries": 0, "duration_seconds": 1.5},
            }]
        }),
        encoding="utf-8",
    )

    db.sync_manifest(manifest)
    assert db.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM metrics WHERE step IS NOT NULL").fetchone()[0] == 4
    assert db.conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM runtime_estimates").fetchone()[0] >= 1
    db.close()


def test_historical_import_is_idempotent(tmp_path):
    from scripts.import_experiment_runs import import_runs

    output = tmp_path / "output"
    run_dir = output / "runs" / "run_test_001"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    assert import_runs(output) == 1
    assert import_runs(output) == 1

    db = ExperimentDB(output / "experiment.db")
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    db.close()
