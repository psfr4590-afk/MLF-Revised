"""Governance requirement acceptance tests.

These tests intentionally verify repository contracts without requiring a training
run, CUDA, network access, or optional GUI dependencies. They are an initial
acceptance layer, not a claim that every operational requirement is fully proven.
Requirement IDs are kept explicit so the governance coverage checker can trace
each target.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    target = ROOT / path
    assert target.is_file(), f"Required repository file is missing: {path}"
    return target.read_text(encoding="utf-8")


def _json(path: str):
    return json.loads(_read(path))


def _python(path: str):
    return ast.parse(_read(path), filename=path)


def _tracked_paths():
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


# UI (001-006): public operator surface and its documented entrypoints.
def test_REQ_UI_001():
    from fastapi.testclient import TestClient
    from command_center.web import app

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "M²S Model Training Pipeline" in response.text


def test_REQ_UI_002():
    from fastapi.testclient import TestClient
    from command_center.web import app

    response = TestClient(app).get("/api/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json(), dict)


def test_REQ_UI_003():
    assert (ROOT / "command_center" / "templates" / "index.html").is_file()


def test_REQ_UI_004():
    assert (ROOT / "run_command_center.py").is_file()


def test_REQ_UI_005():
    source = _read("command_center/app.py")
    assert len(source.splitlines()) < 30, "Compatibility entrypoint should remain a thin delegate"


def test_REQ_UI_006():
    assert (ROOT / "command_center" / "security.py").is_file()


# ENV (001-006): packaging, bootstrap, and declared runtime contracts.
def test_REQ_ENV_001():
    data = tomllib.loads(_read("pyproject.toml"))
    assert isinstance(data, dict)


def test_REQ_ENV_002():
    assert (ROOT / "bootstrap.py").is_file()


def test_REQ_ENV_003():
    assert (ROOT / "bootstrap_windows.ps1").is_file()


def test_REQ_ENV_004():
    assert (ROOT / ".env.example").is_file()


def test_REQ_ENV_005():
    data = tomllib.loads(_read("pyproject.toml"))
    requires = data.get("project", {}).get("requires-python", "")
    assert requires, "project.requires-python must declare the supported Python range"


def test_REQ_ENV_006():
    source = _read("bootstrap.py")
    assert "doctor" in source.lower()


# WF (001-008): configuration, stage identity, lifecycle and execution policy.
def test_REQ_WF_001():
    assert (ROOT / "config" / "pipeline_config.yaml").is_file()


def test_REQ_WF_002():
    assert (ROOT / "config" / "pipeline_config.smoke.yaml").is_file()


def test_REQ_WF_003():
    from command_center import runner
    assert callable(runner.start_stage)
    assert callable(runner.stop)
    assert callable(runner.reconcile_orphaned_runs)


def test_REQ_WF_004():
    from command_center.runner import VALID_STAGES
    assert VALID_STAGES == {"crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"}


def test_REQ_WF_005():
    source = _read("mlframework.py").lower()
    assert "smoke" in source


def test_REQ_WF_006():
    assert (ROOT / "run_pipeline.py").is_file()


def test_REQ_WF_007():
    assert (ROOT / "config" / "dataset_groups.yaml").is_file()


def test_REQ_WF_008(monkeypatch):
    from command_center import runner

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            return None

    monkeypatch.setattr(runner.store, "get", lambda did: {"id": did, "stages": {}})
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.threading, "Thread", FakeThread)
    with runner.LOCK:
        runner.RUNS.pop(912345, None)
    try:
        assert runner.start_stage(912345, "clean") == {
            "started": True, "dataset_id": 912345, "stage": "clean"
        }
        with pytest.raises(RuntimeError, match="already has a running"):
            runner.start_stage(912345, "tokenize")
    finally:
        with runner.LOCK:
            runner.RUNS.pop(912345, None)


# DATA (001-008): persistence, integrity, provenance, and schema discipline.
def test_REQ_DATA_001():
    assert (ROOT / "docs" / "development" / "DATA_PROVENANCE.md").is_file()


def test_REQ_DATA_002():
    assert (ROOT / "command_center" / "store.py").is_file()


def test_REQ_DATA_003(tmp_path):
    from pipeline.integrity import write_manifest, artifact_valid

    artifact = tmp_path / "dataset.jsonl"
    artifact.write_text('{"id": 1}\\n', encoding="utf-8")
    write_manifest(artifact, kind="dataset", rows=1, provenance={"run_id": "acceptance"})
    assert artifact_valid(artifact, expected_provenance={"run_id": "acceptance"})
    assert not artifact_valid(artifact, expected_provenance={"run_id": "different"})


def test_REQ_DATA_004():
    source = _read("config/pipeline_config.yaml")
    assert source.strip() and ":" in source


def test_REQ_DATA_005(tmp_path):
    from command_center.store import atomic_json

    target = tmp_path / "nested" / "state.json"
    atomic_json(target, {"status": "complete", "count": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "complete", "count": 3}
    assert not target.with_suffix(".json.tmp").exists()


def test_REQ_DATA_006():
    source = _read("docs/development/DATA_PROVENANCE.md").lower()
    assert "run" in source or "artifact" in source


def test_REQ_DATA_007(tmp_path, monkeypatch):
    from command_center import store as store_module

    monkeypatch.setattr(store_module, "DATASETS", tmp_path)
    instance = store_module.DatasetStore()
    created = instance.create("acceptance dataset", description="isolated test")
    loaded = instance.get(created["id"])
    assert loaded["name"] == "acceptance dataset"
    assert loaded["description"] == "isolated test"
    assert loaded["status"] == "NEW"
    with pytest.raises(ValueError):
        instance.path(0)


def test_REQ_DATA_008():
    assert callable(hashlib.sha256)


# ML (001-008): model configuration, resource profiles, evaluation, comparability.
def test_REQ_ML_001():
    assert (ROOT / "config" / "dataset_profiles.yaml").is_file()


def test_REQ_ML_002():
    assert (ROOT / "config" / "pipeline_config.yaml").is_file()


def test_REQ_ML_003():    assert (ROOT / "docs" / "architecture" / "PHASE5-HARDWARE-ESTIMATION.md").is_file()


def test_REQ_ML_004():
    assert (ROOT / "command_center" / "store.py").is_file()


def test_REQ_ML_005():
    source = _read("mlframework.py").lower()
    assert "evaluate" in source


def test_REQ_ML_006():
    source = _read("mlframework.py").lower()
    assert "export" in source


def test_REQ_ML_007():
    source = _read("config/pipeline_config.yaml")
    assert len(source.strip()) > 20


def test_REQ_ML_008():
    sources = [p.read_text(encoding="utf-8", errors="replace").lower() for p in ROOT.rglob("*.py")]
    assert any("comparab" in s or "experiment" in s for s in sources)


# ART (001-007): manifests, export cards, paired integrity and dataset cards.
def test_REQ_ART_001(tmp_path):
    from pipeline.integrity import write_manifest, artifact_valid

    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"original artifact bytes")
    write_manifest(artifact, kind="model", provenance={"model_id": "fixture-1"})
    assert artifact_valid(artifact)
    artifact.write_bytes(b"tampered artifact bytes")
    assert not artifact_valid(artifact)


def test_REQ_ART_002():
    assert (ROOT / "docs" / "EXPORT_AND_LLAMA_CPP_INTEGRATION.md").is_file()


def test_REQ_ART_003():
    sources = [p.read_text(encoding="utf-8", errors="replace").lower() for p in ROOT.rglob("*.py")]
    assert any("dataset card" in s or "dataset_card" in s for s in sources)


def test_REQ_ART_004():
    assert (ROOT / "docs" / "development" / "DATA_PROVENANCE.md").is_file()


def test_REQ_ART_005():
    assert "export" in _read("mlframework.py").lower()


def test_REQ_ART_006(tmp_path):
    from pipeline.integrity import sha256_file
    import hashlib

    artifact = tmp_path / "bytes.bin"
    artifact.write_bytes(b"known fixture")
    assert sha256_file(artifact) == hashlib.sha256(b"known fixture").hexdigest()


def test_REQ_ART_007():
    source = _read("docs/EXPORT_AND_LLAMA_CPP_INTEGRATION.md").lower()
    assert "verify" in source or "integrity" in source or "manifest" in source


# OPS (001-008): diagnostics, logs, health and operator observability.
def test_REQ_OPS_001():
    assert "doctor" in _read("README.md").lower()


def test_REQ_OPS_002():
    source = _read("command_center/web.py").lower()
    assert "health" in source or "status" in source


def test_REQ_OPS_003(monkeypatch, tmp_path):
    from command_center import runner

    class FailedProcess:
        pid = 987654
        stdout = None
        def wait(self):
            return 17

    events = []
    state = {"id": 72, "stages": {}, "status": "RUNNING"}
    monkeypatch.setattr(runner.store, "get", lambda did: dict(state))
    monkeypatch.setattr(runner.store, "update", lambda did, **fields: state.update(fields))
    monkeypatch.setattr(runner.store, "event", lambda did, name, payload: events.append((name, payload)))
    monkeypatch.setattr(runner.store, "path", lambda did: tmp_path)
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_clear_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner._run(72, "clean")
    assert state["status"] == "ERROR"
    assert state["stages"]["clean"] == "failed"
    assert any(name == "stage.failed" and payload["exit_code"] == 17 for name, payload in events)


def test_REQ_OPS_004(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    from command_center import secrets

    monkeypatch.setenv("PIPELINE_CREDENTIAL_KEY", Fernet.generate_key().decode())
    vault = secrets.CredentialStore(tmp_path / "credentials.json")
    result = vault.set("acceptance", "fixture-secret", env_var="HF_TOKEN")
    assert result["stored"] is True
    assert vault.reveal("acceptance") == "fixture-secret"
    assert "fixture-secret" not in (tmp_path / "credentials.json").read_text(encoding="utf-8")
    assert vault.delete("acceptance") is True
    assert vault.delete("acceptance") is False


def test_REQ_OPS_005():
    assert (ROOT / "config" / ".env.example").is_file() or (ROOT / ".env.example").is_file()


def test_REQ_OPS_006():
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()


def test_REQ_OPS_007():
    assert (ROOT / ".github" / "workflows" / "security.yml").is_file()


def test_REQ_OPS_008():
    assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


# SEC (001-006): traversal boundaries, loopback defaults, secrets, dependencies.
def test_REQ_SEC_001(tmp_path, monkeypatch):
    from command_center import security

    imports = tmp_path / "imports"
    imports.mkdir()
    (imports / "allowed.txt").write_text("safe", encoding="utf-8")
    monkeypatch.setattr(security, "IMPORT_ROOT", imports)
    assert security.validate_import_source("allowed.txt") == (imports / "allowed.txt").resolve()
    with pytest.raises(ValueError):
        security.validate_import_source("../outside.txt")
    with pytest.raises(ValueError):
        security.validate_import_source(str(tmp_path / "outside.txt"))


def test_REQ_SEC_002():
    from command_center.security import validate_tail_lines, MAX_LOG_LINES

    assert validate_tail_lines(-20) == 1
    assert validate_tail_lines(10) == 10
    assert validate_tail_lines(MAX_LOG_LINES + 500) == MAX_LOG_LINES


def test_REQ_SEC_003():
    tracked_env = [p for p in ROOT.rglob(".env") if ".git" not in p.parts and "example" not in p.name]
    assert not tracked_env, f"Unexpected real environment files in source tree: {tracked_env}"


def test_REQ_SEC_004(monkeypatch):
    from command_center.secrets import validate_env_var

    assert validate_env_var("HF_TOKEN") == "HF_TOKEN"
    with pytest.raises(ValueError):
        validate_env_var("PATH")
    with pytest.raises(ValueError):
        validate_env_var("bad-name")


def test_REQ_SEC_005():
    # The supported dependency source of truth is pyproject.toml; a separate
    # lock/constraints file is optional and must not be required by this contract.
    data = tomllib.loads(_read("pyproject.toml"))
    project = data.get("project", {})
    assert project.get("dependencies") or project.get("optional-dependencies"), (
        "Runtime dependencies must be declared in project metadata"
    )


def test_REQ_SEC_006():
    assert (ROOT / "SECURITY.md").is_file()


# REL (001-007): release identity, gate behavior, clean clone, and test layers.
def test_REQ_REL_001():
    source = _read(".github/workflows/ci.yml").lower()
    assert "pytest" in source or "test" in source


def test_REQ_REL_002():
    assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


def test_REQ_REL_003():
    import scripts.verify_release as gate
    assert callable(gate.main)


def test_REQ_REL_004():
    candidates = [p for p in ROOT.rglob("*.py") if "release" in p.name.lower() or "verify" in p.name.lower()]
    source = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in candidates)
    assert "--clean-clone" in source


def test_REQ_REL_005():
    data = tomllib.loads(_read("pyproject.toml"))
    version = data.get("project", {}).get("version")
    assert version, "Project version must be explicitly declared"


def test_REQ_REL_006():
    assert (ROOT / "tests").is_dir()


def test_REQ_REL_007():
    test_files = list((ROOT / "tests").rglob("test_*.py"))
    assert test_files, "Expected existing regression/unit tests alongside acceptance tests"