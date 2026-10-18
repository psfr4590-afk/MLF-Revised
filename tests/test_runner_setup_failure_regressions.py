"""Regression tests for failures during runner setup."""
from __future__ import annotations


def test_worker_setup_failure_clears_runner_reservation_and_state(monkeypatch):
    from command_center import runner

    did = 982321
    dataset = {
        "id": did,
        "stages": {"clean": "pending"},
        "status": "NEW",
    }
    events = []
    cleared = []

    def update(_did, **fields):
        dataset.update(fields)

    def fail_during_setup(_did):
        raise RuntimeError("simulated worker setup failure")

    monkeypatch.setattr(runner.store, "get", lambda _did: dict(dataset))
    monkeypatch.setattr(runner.store, "update", update)
    monkeypatch.setattr(
        runner.store,
        "event",
        lambda _did, name, payload: events.append((name, payload)),
    )
    monkeypatch.setattr(runner.store, "path", fail_during_setup)
    monkeypatch.setattr(runner, "_clear_state", lambda _did: cleared.append(_did))

    with runner.LOCK:
        runner.RUNS.pop(did, None)
        runner.CANCEL_REQUESTED.discard(did)
        runner.RUNS[did] = runner._STARTING

    try:
        runner._run(did, "clean")

        with runner.LOCK:
            assert did not in runner.RUNS
            assert did not in runner.CANCEL_REQUESTED

        assert dataset["status"] == "ERROR"
        assert dataset["stages"]["clean"] == "failed"
        assert cleared == [did]
        assert any(
            name == "stage.setup_failed"
            and payload.get("stage") == "clean"
            and payload.get("error_type") == "RuntimeError"
            for name, payload in events
        )
        assert "simulated worker setup failure" not in repr(events)
    finally:
        with runner.LOCK:
            runner.RUNS.pop(did, None)
            runner.CANCEL_REQUESTED.discard(did)


def test_worker_error_preserves_terminal_stage_result(monkeypatch):
    from command_center import runner

    did = 982322
    dataset = {
        "id": did,
        "stages": {"clean": "complete"},
        "status": "COMPLETE",
    }
    events = []
    cleared = []

    def update(_did, **fields):
        dataset.update(fields)

    def fail_after_completion(_did, _stage):
        raise RuntimeError("simulated post-completion failure")

    monkeypatch.setattr(runner.store, "get", lambda _did: dict(dataset))
    monkeypatch.setattr(runner.store, "update", update)
    monkeypatch.setattr(
        runner.store,
        "event",
        lambda _did, name, payload: events.append((name, payload)),
    )
    monkeypatch.setattr(runner, "_clear_state", lambda _did: cleared.append(_did))
    monkeypatch.setattr(runner, "_run_inner", fail_after_completion)

    with runner.LOCK:
        runner.RUNS.pop(did, None)
        runner.CANCEL_REQUESTED.discard(did)
        runner.RUNS[did] = runner._STARTING

    try:
        runner._run(did, "clean")

        with runner.LOCK:
            assert did not in runner.RUNS
            assert did not in runner.CANCEL_REQUESTED

        assert dataset["stages"]["clean"] == "complete"
        assert dataset["status"] == "COMPLETE"
        assert cleared == [did]
        assert any(
            name == "stage.worker_error"
            and payload.get("stage") == "clean"
            and payload.get("error_type") == "RuntimeError"
            for name, payload in events
        )
        assert "simulated post-completion failure" not in repr(events)
    finally:
        with runner.LOCK:
            runner.RUNS.pop(did, None)
            runner.CANCEL_REQUESTED.discard(did)
