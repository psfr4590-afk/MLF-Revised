"""Behavioral regression tests for runner start/cancel lifecycle semantics."""
from __future__ import annotations

import threading


def test_stop_records_cancellation_while_worker_is_starting(monkeypatch):
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
    monkeypatch.setattr(runner.store, "refresh_pipeline_state", lambda _did: (_ for _ in ()).throw(AssertionError("cancellation state must not be overwritten")))
    monkeypatch.setattr(runner.store, "refresh_stats", lambda _did: None)
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_clear_state", lambda _did: None)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.credentials, "environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cancelled worker must not spawn")),
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


def test_concurrent_start_reservation_rejects_duplicate_launch(monkeypatch, tmp_path):
    from command_center import runner

    did = 982303
    dataset = {"id": did, "stages": {"train": "pending"}, "status": "NEW"}
    monkeypatch.setattr(runner.store, "get", lambda _did: dataset)
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_clear_state", lambda _did: None)
    started = threading.Event()

    class BlockedThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            started.set()

    monkeypatch.setattr(runner.threading, "Thread", BlockedThread)
    with runner.LOCK:
        runner.RUNS.pop(did, None)
        runner.CANCEL_REQUESTED.discard(did)
    try:
        assert runner.start_stage(did, "train")["started"] is True
        assert started.is_set()
        try:
            runner.start_stage(did, "train")
        except RuntimeError as exc:
            assert "already has a running or starting" in str(exc)
        else:
            raise AssertionError("duplicate stage launch was accepted")
    finally:
        with runner.LOCK:
            runner.RUNS.pop(did, None)
            runner.CANCEL_REQUESTED.discard(did)
