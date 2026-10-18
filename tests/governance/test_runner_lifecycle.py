"""Behavioral tests for runner cancellation, launch cleanup, and process trees."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_stop_records_cancellation_while_worker_is_starting():
    from command_center import runner

    did = 982301
    with runner.LOCK:
        runner.RUNS[did] = runner._STARTING
        runner.CANCEL_REQUESTED.discard(did)
    try:
        assert runner.stop(did) is True
        assert did in runner.CANCEL_REQUESTED
    finally:
        with runner.LOCK:
            runner.RUNS.pop(did, None)
            runner.CANCEL_REQUESTED.discard(did)


def test_worker_honors_cancel_before_spawning_subprocess(monkeypatch, tmp_path):
    from command_center import runner

    did = 982302
    dataset = {"id": did, "stages": {"clean": "pending"}, "status": "NEW"}
    events = []
    monkeypatch.setattr(runner.store, "get", lambda _did: dict(dataset))
    monkeypatch.setattr(runner.store, "update", lambda _did, **fields: dataset.update(fields))
    monkeypatch.setattr(runner.store, "event", lambda _did, name, payload: events.append((name, payload)))
    monkeypatch.setattr(runner.store, "path", lambda _did: tmp_path)
    monkeypatch.setattr(runner.store, "refresh_pipeline_state", lambda _did: pytest.fail("cancelled state was overwritten"))
    monkeypatch.setattr(runner.store, "refresh_stats", lambda _did: None)
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_clear_state", lambda _did: None)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.credentials, "environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("cancelled worker must not spawn"),
    )
    with runner.LOCK:
        runner.RUNS[did] = runner._STARTING
        runner.CANCEL_REQUESTED.add(did)

    runner._run(did, "clean")

    assert dataset["status"] == "CANCELLED"
    assert dataset["stages"]["clean"] == "cancelled"
    assert any(name == "stage.cancelled" and payload["phase"] == "before_launch" for name, payload in events)
    with runner.LOCK:
        assert did not in runner.RUNS
        assert did not in runner.CANCEL_REQUESTED


def test_start_stage_clears_reservation_when_state_write_fails(monkeypatch):
    from command_center import runner

    did = 982303
    monkeypatch.setattr(runner.store, "get", lambda _did: {"id": did, "stages": {}})
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failure")))
    cleared = []
    monkeypatch.setattr(runner, "_clear_state", lambda _did: cleared.append(_did))
    with runner.LOCK:
        runner.RUNS.pop(did, None)
        runner.CANCEL_REQUESTED.discard(did)
    with pytest.raises(OSError, match="disk failure"):
        runner.start_stage(did, "clean")
    with runner.LOCK:
        assert did not in runner.RUNS
        assert did not in runner.CANCEL_REQUESTED
    assert cleared == [did]


def test_windows_termination_targets_the_entire_process_tree(monkeypatch):
    from command_center import runner

    class Process:
        pid = 765432

        def __init__(self):
            self.wait_calls = []

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            return 0

    process = Process()
    calls = []
    monkeypatch.setattr(runner.os, "name", "nt")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: (calls.append((command, kwargs)) or SimpleNamespace(returncode=0)),
    )

    runner._terminate_process_tree(process)

    assert calls[0][0][:2] == ["taskkill", "/PID"]
    assert calls[0][0][2:] == [str(process.pid), "/T", "/F"]
    assert process.wait_calls == [runner.STOP_TIMEOUT_SECONDS]
