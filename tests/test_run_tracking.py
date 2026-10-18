from __future__ import annotations

import json
from pathlib import Path

from pipeline.run_tracking import RunTracker


def _patch_host(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.run_tracking._git_state", lambda root: {
        "commit": "abc123", "branch": "phase3", "dirty": False, "status_entries": []
    })
    monkeypatch.setattr("pipeline.run_tracking._hardware", lambda: {"cpu": "test"})
    monkeypatch.setattr("pipeline.run_tracking._torch_environment", lambda: {"cuda_available": False})


def test_run_manifest_has_single_authoritative_identity(tmp_path: Path, monkeypatch) -> None:
    _patch_host(monkeypatch)
    tracker = RunTracker(
        tmp_path, tmp_path / "output", run_id="run_test_001",
        dataset_id="dataset_001", experiment_id="exp_test_001",
        configuration={"train": {"total_steps": 10}},
        dataset_identity={"catalog_sha256": "abc"},
        source_identity={"sources": [{"kind": "web"}]},
    )
    data = json.loads(tracker.path.read_text(encoding="utf-8"))
    assert tracker.path == tmp_path / "output" / "runs" / "run_test_001" / "run_manifest.json"
    assert data["run_id"] == "run_test_001"
    assert data["dataset_id"] == "dataset_001"
    assert data["experiment_id"] == "exp_test_001"
    assert data["git"]["commit"] == "abc123"
    assert data["final_status"] == "RUNNING"


def test_stage_record_captures_lifecycle_inputs_outputs_and_hashes(tmp_path: Path, monkeypatch) -> None:
    _patch_host(monkeypatch)
    source = tmp_path / "input.txt"
    output = tmp_path / "output.txt"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    tracker = RunTracker(tmp_path, tmp_path / "output", run_id="run_stage", dataset_id="ds", experiment_id="exp", configuration={})
    tracker.start_stage("clean", inputs=[source])
    tracker.finish_stage("clean", outputs=[output], warnings=["test warning"], metrics={"rows": 1})
    data = json.loads(tracker.path.read_text(encoding="utf-8"))
    stage = data["stages"]["clean"]
    assert stage["status"] == "PASS"
    assert stage["end"]
    assert stage["duration_seconds"] >= 0
    assert stage["inputs"][0]["sha256"]
    assert stage["outputs"][0]["sha256"]
    assert stage["artifact_hashes"]["output.txt"] == stage["outputs"][0]["sha256"]
    assert data["warnings"] == ["test warning"]
    assert data["metrics"]["clean"]["rows"] == 1


def test_failed_stage_and_run_are_durable(tmp_path: Path, monkeypatch) -> None:
    _patch_host(monkeypatch)
    tracker = RunTracker(tmp_path, tmp_path / "output", run_id="run_fail", dataset_id="ds", experiment_id="exp", configuration={})
    tracker.start_stage("train")
    tracker.fail_stage("train", RuntimeError("boom"))
    tracker.finish_run("FAILED")
    data = json.loads(tracker.path.read_text(encoding="utf-8"))
    assert data["stages"]["train"]["status"] == "FAILED"
    assert data["errors"] == ["RuntimeError: boom"]
    assert data["final_status"] == "FAILED"


def test_resume_reuses_same_run_manifest(tmp_path: Path, monkeypatch) -> None:
    _patch_host(monkeypatch)
    tracker = RunTracker(tmp_path, tmp_path / "output", run_id="run_resume", dataset_id="ds", experiment_id="exp", configuration={})
    tracker.start_stage("train")
    tracker.finish_stage("train")
    resumed = RunTracker.resume(tmp_path, tmp_path / "output", tracker.path)
    assert resumed.run_id == tracker.run_id
    assert resumed.manifest["final_status"] == "RUNNING"


def test_finish_run_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    _patch_host(monkeypatch)
    tracker = RunTracker(tmp_path, tmp_path / "output", run_id="run_once", dataset_id="ds", experiment_id="exp", configuration={})
    tracker.finish_run("PASS")
    first = json.loads(tracker.path.read_text(encoding="utf-8"))
    ended = first["timestamps"]["ended"]
    tracker.finish_run("FAILED")
    second = json.loads(tracker.path.read_text(encoding="utf-8"))
    assert second["final_status"] == "PASS"
    assert second["timestamps"]["ended"] == ended


# Regression coverage for the persisted run-id pointer.
def test_fresh_execution_never_reuses_persisted_run_id(tmp_path, monkeypatch):
    import json
    from pipeline.orchestrator import _load_or_create_run_id

    output = tmp_path / "output"
    run_id_path = output / "provenance" / "run_id.txt"
    run_id_path.parent.mkdir(parents=True)
    run_id_path.write_text("old-run\n", encoding="utf-8")
    manifest = output / "runs" / "old-run" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_id": "old-run", "final_status": "RUNNING", "timestamps": {"ended": None}}), encoding="utf-8")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "new-run")

    assert _load_or_create_run_id(output, resume=False) == "new-run"
    assert run_id_path.read_text(encoding="utf-8").strip() == "new-run"
    assert manifest.is_file()


def test_resume_reuses_only_an_unfinished_run(tmp_path, monkeypatch):
    import json
    from pipeline.orchestrator import _load_or_create_run_id

    output = tmp_path / "output"
    run_id_path = output / "provenance" / "run_id.txt"
    run_id_path.parent.mkdir(parents=True)
    run_id_path.write_text("active-run\n", encoding="utf-8")
    manifest = output / "runs" / "active-run" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_id": "active-run", "final_status": "RUNNING", "timestamps": {"ended": None}}), encoding="utf-8")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "fresh-run")

    assert _load_or_create_run_id(output, resume=True) == "active-run"


def test_resume_rotates_completed_or_stale_run_id(tmp_path, monkeypatch):
    import json
    from pipeline.orchestrator import _load_or_create_run_id

    output = tmp_path / "output"
    run_id_path = output / "provenance" / "run_id.txt"
    run_id_path.parent.mkdir(parents=True)
    run_id_path.write_text("completed-run\n", encoding="utf-8")
    manifest = output / "runs" / "completed-run" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_id": "completed-run", "final_status": "PASS", "timestamps": {"ended": "finished"}}), encoding="utf-8")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "fresh-run")

    assert _load_or_create_run_id(output, resume=True) == "fresh-run"
    assert run_id_path.read_text(encoding="utf-8").strip() == "fresh-run"


def test_tokenizer_stage_records_the_materialized_artifacts(tmp_path, monkeypatch):
    from pipeline.orchestrator import Pipeline

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = tokenizer_dir / "tokenizer.json"
    tokenizer.write_text('{"model":{}}', encoding="utf-8")
    tokenizer_manifest = Path(str(tokenizer) + ".manifest.json")
    tokenizer_manifest.write_text('{"sha256":"fixture"}', encoding="utf-8")

    class Tracker:
        def __init__(self):
            self.manifest = {}
            self.outputs = None
        def start_stage(self, name, *, inputs=None):
            self.name = name
        def finish_stage(self, name, *, outputs=None, metrics=None):
            self.outputs = outputs
        def fail_stage(self, name, exc):
            raise exc

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.cfg = {"tokenizer": {"output_path": str(tokenizer_dir)}}
    pipeline._out = tmp_path / "output"
    pipeline._stage_metrics = {}
    pipeline._source_observability = []
    pipeline._run_tracker = Tracker()
    result = object()

    assert pipeline._run_stage("tokenize", lambda: result) is result
    assert pipeline._run_tracker.outputs == [tokenizer, tokenizer_manifest]
