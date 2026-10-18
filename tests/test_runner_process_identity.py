from __future__ import annotations

from types import SimpleNamespace

import command_center.runner as runner


def _fake_psutil(create_time):
    return SimpleNamespace(
        Process=lambda pid: SimpleNamespace(create_time=lambda: create_time),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
    )


def test_dead_pid_is_not_the_same_process(monkeypatch):
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: False)

    assert runner._same_process_identity(
        {"pid": 123, "process_identity": {"pid": 123, "create_time": 100.0}}
    ) is False


def test_matching_creation_time_preserves_original_worker(monkeypatch):
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(runner, "psutil", _fake_psutil(100.0))

    assert runner._same_process_identity(
        {"pid": 123, "process_identity": {"pid": 123, "create_time": 100.0}}
    ) is True


def test_reused_pid_with_different_creation_time_is_rejected(monkeypatch):
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(runner, "psutil", _fake_psutil(200.0))

    assert runner._same_process_identity(
        {"pid": 123, "process_identity": {"pid": 123, "create_time": 100.0}}
    ) is False


def test_missing_identity_evidence_preserves_live_pid(monkeypatch):
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(runner, "psutil", None)

    assert runner._same_process_identity({"pid": 123, "process_identity": None}) is True
    assert runner._same_process_identity({"pid": 123}) is True
