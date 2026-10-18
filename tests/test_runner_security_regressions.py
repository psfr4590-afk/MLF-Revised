"""Security regression tests for runner logging boundaries."""
from __future__ import annotations


def test_runner_redacts_inherited_environment_secrets_in_common_encodings():
    from command_center.runner import _redact_text, _sensitive_environment_values

    secret = "s3cr3t/token+value"
    values = _sensitive_environment_values({
        "GITHUB_TOKEN": secret,
        "SERVICE_PASSWORD": "pw-value",
        "NORMAL_SETTING": "visible-value",
        "EMPTY_API_KEY": "",
    })

    assert secret in values
    assert "pw-value" in values
    assert "visible-value" not in values
    output = _redact_text(
        f"raw={secret} url={__import__('urllib.parse', fromlist=['quote']).quote(secret, safe='')} "
        f"b64={__import__('base64').b64encode(secret.encode()).decode()}",
        values,
    )
    assert secret not in output
    assert __import__("urllib.parse", fromlist=["quote"]).quote(secret, safe="") not in output
    assert __import__("base64").b64encode(secret.encode()).decode() not in output


def test_runner_launch_failure_event_does_not_include_exception_message(monkeypatch, tmp_path):
    from command_center import runner

    did = 982311
    secret = "canary-secret-do-not-log"
    dataset = {"id": did, "stages": {"clean": "pending"}, "status": "NEW"}
    events = []
    monkeypatch.setattr(runner.store, "get", lambda _did: dict(dataset))
    monkeypatch.setattr(runner.store, "update", lambda _did, **fields: dataset.update(fields))
    monkeypatch.setattr(runner.store, "event", lambda _did, name, payload: events.append((name, payload)))
    monkeypatch.setattr(runner.store, "path", lambda _did: tmp_path)
    monkeypatch.setattr(runner.store, "refresh_stats", lambda _did: None)
    monkeypatch.setattr(runner, "_write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_clear_state", lambda _did: None)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.credentials, "environment", lambda: {})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError(secret)))

    with runner.LOCK:
        runner.RUNS[did] = runner._STARTING
        runner.CANCEL_REQUESTED.discard(did)
    try:
        runner._run(did, "clean")
        payload = next(payload for name, payload in events if name == "stage.launch_failed")
        assert payload == {"stage": "clean", "error_type": "OSError"}
        assert secret not in repr(events)
        assert dataset["status"] == "ERROR"
    finally:
        with runner.LOCK:
            runner.RUNS.pop(did, None)
            runner.CANCEL_REQUESTED.discard(did)
