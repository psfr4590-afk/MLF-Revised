"""Supplemental operational failure-mode tests for governance hardening.

These tests complement the 64 catalog-mapped targets without changing their
traceability identities. All filesystem effects are isolated in tmp_path.
"""
from __future__ import annotations

import json

import pytest


def test_runner_stop_delegates_active_worker_termination(monkeypatch):
    from command_center import runner

    class Process:
        pid = 991101

        def poll(self):
            return None

    process = Process()
    terminated = []
    monkeypatch.setattr(runner, "_terminate_process_tree", lambda proc: terminated.append(proc))
    with runner.LOCK:
        runner.RUNS[991101] = process
        runner.CANCEL_REQUESTED.discard(991101)
    try:
        assert runner.stop(991101) is True
        assert terminated == [process]
        assert 991101 in runner.CANCEL_REQUESTED
    finally:
        with runner.LOCK:
            runner.RUNS.pop(991101, None)
            runner.CANCEL_REQUESTED.discard(991101)


def test_runner_stop_requests_cancellation_of_starting_worker():
    from command_center import runner

    with runner.LOCK:
        runner.RUNS[991102] = runner._STARTING
        runner.CANCEL_REQUESTED.discard(991102)
    try:
        assert runner.stop(991102) is True
        assert 991102 in runner.CANCEL_REQUESTED
    finally:
        with runner.LOCK:
            runner.RUNS.pop(991102, None)
            runner.CANCEL_REQUESTED.discard(991102)


def test_orphan_recovery_marks_dead_worker_failed(monkeypatch, tmp_path):
    from command_center import runner

    dataset = {"id": 991103, "status": "RUNNING", "stages": {"train": "running"}}
    state_path = tmp_path / "runner_state.json"
    state_path.write_text(json.dumps({"pid": 991104, "stage": "train"}), encoding="utf-8")
    events = []
    updates = []
    monkeypatch.setattr(runner.store, "list", lambda: [dataset])
    monkeypatch.setattr(runner, "_state_path", lambda did: state_path)
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(runner.store, "update", lambda did, **fields: updates.append((did, fields)))
    monkeypatch.setattr(runner.store, "event", lambda *args: events.append(args))
    monkeypatch.setattr(runner, "_clear_state", lambda did: state_path.unlink(missing_ok=True))

    runner.reconcile_orphaned_runs()

    assert updates[0][1]["status"] == "ERROR"
    assert updates[0][1]["stages"]["train"] == "failed"
    assert events[0][1] == "stage.recovered_orphan"


def test_orphan_recovery_preserves_live_worker(monkeypatch, tmp_path):
    from command_center import runner

    dataset = {"id": 991105, "status": "RUNNING", "stages": {"train": "running"}}
    state_path = tmp_path / "runner_state.json"
    state_path.write_text(json.dumps({"pid": 991106, "stage": "train"}), encoding="utf-8")
    updates = []
    monkeypatch.setattr(runner.store, "list", lambda: [dataset])
    monkeypatch.setattr(runner, "_state_path", lambda did: state_path)
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(runner.store, "update", lambda *args, **kwargs: updates.append(args))
    monkeypatch.setattr(runner.store, "event", lambda *args: pytest.fail("live worker must not be recovered as orphan"))

    runner.reconcile_orphaned_runs()
    assert updates == []
    assert state_path.exists()


def test_artifact_provenance_mismatch_and_manifest_corruption_fail_closed(tmp_path):
    from pipeline.integrity import write_manifest, artifact_valid, manifest_path

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"valid artifact bytes")
    write_manifest(artifact, kind="model", provenance={"run_id": "run-a", "config": "cfg-a"})
    assert artifact_valid(artifact, expected_provenance={"run_id": "run-a", "config": "cfg-a"})
    assert not artifact_valid(artifact, expected_provenance={"run_id": "run-b", "config": "cfg-a"})
    manifest_path(artifact).write_text("{invalid", encoding="utf-8")
    assert not artifact_valid(artifact)


def test_source_manifest_rejects_dataset_group_mismatch(tmp_path):
    from scripts.verify_release import _validate_source_manifest

    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps({
        "schema": 2, "run_id": "fixture", "dataset_group": "group-a",
        "retrieval_started_at": "2026-01-01T00:00:00+00:00",
        "retrieval_completed_at": "2026-01-01T00:00:01+00:00",
        "source_definition_sha256": "0" * 64,
        "source_definition_files": {},
        "sources": [{"kind": "local_fixture", "identifier": "x", "revision": "1",
                     "license": "test", "raw_source_sha256": None,
                     "dataset_group": "group-b"}],
        "rights_note": "test fixture"
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset-group identity"):
        _validate_source_manifest(path)


def test_release_gate_rejects_rc_without_required_flags(monkeypatch):
    from scripts import verify_release

    monkeypatch.setattr("sys.argv", ["verify_release.py", "--profile", "rc"])
    with pytest.raises(SystemExit) as exc:
        verify_release.main()
    assert exc.value.code == 2


def test_release_gate_rejects_skip_tests(monkeypatch):
    from scripts import verify_release

    monkeypatch.setattr("sys.argv", ["verify_release.py", "--skip-tests"])
    assert verify_release.main() == 2


def test_project_version_matches_declared_package_version():
    import tomllib
    from scripts.verify_release import _project_version, ROOT

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert _project_version() == declared


def test_worker_log_redacts_injected_credential_value(monkeypatch, tmp_path):
    from command_center import runner

    secret = "fixture-secret-value-123"
    log_path = tmp_path / "logs" / "command-center.log"
    class FakeProcess:
        pid = 123456
        stdout = iter([f"starting worker with token={secret}\\n"])
        def wait(self):
            return 1

    dataset = {"id": 991107, "stages": {}, "status": "NEW"}
    monkeypatch.setattr(runner.store, "get", lambda did: dict(dataset))
    def update(did, **fields):
        dataset.update(fields)
    monkeypatch.setattr(runner.store, "update", update)
    monkeypatch.setattr(runner.store, "event", lambda *args: None)
    monkeypatch.setattr(runner.store, "path", lambda did: tmp_path)
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_clear_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runner, "credentials", type("Credentials", (), {
        "environment": staticmethod(lambda: {"HF_TOKEN": secret})
    })())
    monkeypatch.setattr(runner.store, "refresh_stats", lambda did: None)
    monkeypatch.setattr(runner.store, "refresh_pipeline_state", lambda did: None)

    runner._run(991107, "clean")
    output = log_path.read_text(encoding="utf-8")
    assert "[REDACTED]" in output
    assert secret not in output


def test_stop_escalates_and_waits_for_process_tree(monkeypatch):
    from command_center import runner
    import signal
    import subprocess

    class Process:
        pid = 765432

        def __init__(self):
            self.killed = False
            self.terminated = False
            self.wait_calls = 0

        def poll(self):
            return None if not self.killed else -getattr(signal, "SIGKILL", 9)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -getattr(signal, "SIGKILL", 9)

    proc = Process()
    if runner.os.name == "posix":
        sent_signals = []

        def fake_killpg(pid, sig):
            assert pid == proc.pid
            sent_signals.append(sig)
            if sig == signal.SIGKILL:
                proc.killed = True

        monkeypatch.setattr(runner.os, "killpg", fake_killpg)
    with runner.LOCK:
        runner.RUNS[991108] = proc
        runner.CANCEL_REQUESTED.discard(991108)
    try:
        assert runner.stop(991108) is True
        if runner.os.name == "posix":
            assert sent_signals == [signal.SIGTERM, signal.SIGKILL]
        else:
            # Windows currently guarantees direct-child termination only;
            # process-tree containment via Job Objects remains a release blocker.
            assert proc.terminated
            assert proc.killed
        assert proc.wait_calls >= 2
    finally:
        with runner.LOCK:
            runner.RUNS.pop(991108, None)
            runner.CANCEL_REQUESTED.discard(991108)


def test_redaction_handles_short_secrets_and_common_encodings():
    import base64
    from command_center.runner import _redact_text

    secret = "xY9"
    encoded = base64.b64encode(secret.encode()).decode()
    line = " ".join([secret, encoded, __import__("urllib.parse", fromlist=["quote"]).quote(secret),
                     __import__("json").dumps(secret)])
    cleaned = _redact_text(line, [secret])
    assert secret not in cleaned
    assert encoded not in cleaned
    assert __import__("json").dumps(secret) not in cleaned


def test_redaction_sanitizes_exception_text_and_structured_fields():
    import json
    from command_center.runner import _redact_text

    secret = "exception-secret-456"
    message = json.dumps({"event": "worker.error", "message": f"request failed token={secret}",
                          "exception": f"RuntimeError({secret})"})
    cleaned = _redact_text(message, [secret])
    assert secret not in cleaned
    assert "[REDACTED]" in cleaned


def test_restart_recovery_does_not_clear_live_process_state(monkeypatch, tmp_path):
    from command_center import runner

    state_path = tmp_path / "runner_state.json"
    state_path.write_text(json.dumps({"schema": 1, "dataset_id": 991109, "stage": "train",
                                      "pid": 991110, "status": "running"}), encoding="utf-8")
    dataset = {"id": 991109, "status": "RUNNING", "stages": {"train": "running"}}
    monkeypatch.setattr(runner.store, "list", lambda: [dataset])
    monkeypatch.setattr(runner, "_state_path", lambda did: state_path)
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(runner.store, "update", lambda *args, **kwargs: pytest.fail("must preserve live run"))
    monkeypatch.setattr(runner.store, "event", lambda *args: pytest.fail("must not emit orphan event"))
    runner.reconcile_orphaned_runs()
    assert state_path.exists()
