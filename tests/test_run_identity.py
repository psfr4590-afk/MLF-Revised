from __future__ import annotations

import json
from pathlib import Path

from pipeline.orchestrator import _load_or_create_run_id


def _persisted_run(output: Path, run_id: str, *, final_status: str = "RUNNING", ended=None) -> None:
    run_id_path = output / "provenance" / "run_id.txt"
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.write_text(run_id + "\n", encoding="utf-8")
    manifest_path = output / "runs" / run_id / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "run_id": run_id,
        "final_status": final_status,
        "timestamps": {"started": "start", "updated": "update", "ended": ended},
    }), encoding="utf-8")


def test_fresh_execution_never_reuses_persisted_run_id(tmp_path, monkeypatch):
    output = tmp_path / "output"
    _persisted_run(output, "old-run", final_status="RUNNING")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "new-run")

    run_id = _load_or_create_run_id(output, resume=False)

    assert run_id == "new-run"
    assert (output / "provenance" / "run_id.txt").read_text(encoding="utf-8").strip() == "new-run"
    assert (output / "runs" / "old-run" / "run_manifest.json").is_file()


def test_resume_reuses_only_an_unfinished_run(tmp_path, monkeypatch):
    output = tmp_path / "output"
    _persisted_run(output, "active-run", final_status="RUNNING")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "fresh-run")

    assert _load_or_create_run_id(output, resume=True) == "active-run"


def test_resume_rotates_completed_run_id(tmp_path, monkeypatch):
    output = tmp_path / "output"
    _persisted_run(output, "completed-run", final_status="PASS", ended="finished")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "fresh-run")

    assert _load_or_create_run_id(output, resume=True) == "fresh-run"
    assert (output / "provenance" / "run_id.txt").read_text(encoding="utf-8").strip() == "fresh-run"


def test_resume_rotates_stale_id_without_manifest(tmp_path, monkeypatch):
    output = tmp_path / "output"
    run_id_path = output / "provenance" / "run_id.txt"
    run_id_path.parent.mkdir(parents=True)
    run_id_path.write_text("stale-run\n", encoding="utf-8")
    monkeypatch.setattr("pipeline.orchestrator.new_run_id", lambda: "fresh-run")

    assert _load_or_create_run_id(output, resume=True) == "fresh-run"
