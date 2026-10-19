from __future__ import annotations

import json
from pathlib import Path

import pipeline.readiness as readiness


def _doctor_checks(_root):
    return True, [
        {"name": "Python", "ok": True, "detail": "3.12", "required": True},
        {"name": "output writable", "ok": True, "detail": "writable", "required": True},
        {"name": "optional tool", "ok": False, "detail": "not installed", "required": False},
    ]


def _hardware():
    class Profile:
        gpu_count = 0
        def to_dict(self):
            return {"platform": "test", "gpu_count": 0, "gpu_memory_gb": 0.0, "cuda_available": False}
    return Profile()


def _training_profile(_hardware):
    class Profile:
        def to_dict(self):
            return {"model_preset": "85M", "seq_len": 256, "batch_size": 1,
                    "precision": "fp32", "decision_reasons": ["CPU"], "reason": "CPU"}
    return Profile()


def test_readiness_reports_resource_checks_and_no_benchmark_by_default(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("pipeline.doctor.run_doctor", _doctor_checks)
    monkeypatch.setattr("pipeline.model_sizer.profile_hardware", lambda _root: _hardware())
    monkeypatch.setattr("pipeline.model_sizer.recommend_training_profile", _training_profile)
    monkeypatch.setattr(readiness, "_available_ram_gib", lambda: 2.0)
    monkeypatch.setattr(readiness, "_disk_state", lambda _path: {"path": str(tmp_path), "free_gib": 20.0, "total_gib": 100.0, "available": True})
    monkeypatch.setattr("scripts.system_info._torch_state", lambda: {"installed": True, "version": "test", "cuda_available": False, "device_count": 0})
    monkeypatch.setattr("scripts.system_info._nvidia_state", lambda: {"available": False})
    monkeypatch.setattr(readiness, "_git_identity", lambda _root: {"commit": "abc", "branch": "test"})

    report = readiness.build_readiness_report(tmp_path)

    assert report["summary"]["status"] == "READY_WITH_LIMITS"
    assert report["memory"]["available_ram_gib"] == 2.0
    assert report["training_profile"]["model_preset"] == "85M"
    assert report["benchmark"]["status"] == "NOT_RUN"
    assert any(item["name"] == "Available system RAM" and item["status"] == "WARN" for item in report["checks"])


def test_required_failure_blocks_but_optional_failure_only_warns(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("pipeline.doctor.run_doctor", lambda _root: (False, [
        {"name": "required", "ok": False, "detail": "broken", "required": True},
        {"name": "optional", "ok": False, "detail": "absent", "required": False},
    ]))
    monkeypatch.setattr("pipeline.model_sizer.profile_hardware", lambda _root: _hardware())
    monkeypatch.setattr("pipeline.model_sizer.recommend_training_profile", _training_profile)
    monkeypatch.setattr(readiness, "_available_ram_gib", lambda: 12.0)
    monkeypatch.setattr(readiness, "_disk_state", lambda _path: {"path": str(tmp_path), "free_gib": 20.0, "total_gib": 100.0, "available": True})
    monkeypatch.setattr("scripts.system_info._torch_state", lambda: {"installed": True, "version": "test", "cuda_available": False, "device_count": 0})
    monkeypatch.setattr("scripts.system_info._nvidia_state", lambda: {"available": False})
    monkeypatch.setattr(readiness, "_git_identity", lambda _root: {"commit": None, "branch": None})

    report = readiness.build_readiness_report(tmp_path)

    assert report["summary"]["status"] == "BLOCKED"
    assert report["summary"]["blockers"] == 1


def test_write_report_is_atomic_json(tmp_path: Path):
    target = tmp_path / "runtime" / "readiness.json"
    report = {"schema_version": 1, "summary": {"status": "READY"}}

    written = readiness.write_readiness_report(report, target)

    assert json.loads(written.read_text(encoding="utf-8")) == report
    assert not target.with_suffix(".json.tmp").exists()
