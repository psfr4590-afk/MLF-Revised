from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet


def test_legacy_cli_imports_cleanly():
    import command_center.cli as cli
    assert callable(cli.main)


def test_credential_environment_allowlist():
    from command_center.secrets import validate_env_var

    assert validate_env_var("GITHUB_TOKEN") == "GITHUB_TOKEN"
    assert validate_env_var("") == ""
    with pytest.raises(ValueError, match="Unsupported credential environment variable"):
        validate_env_var("PYTHONPATH")


def test_credential_store_rejects_unsafe_environment_variable(tmp_path: Path, monkeypatch):
    from command_center.secrets import CredentialStore

    monkeypatch.setenv("PIPELINE_CREDENTIAL_KEY", Fernet.generate_key().decode())
    store = CredentialStore(tmp_path / "credentials.json")
    with pytest.raises(ValueError, match="Unsupported credential environment variable"):
        store.set("bad", "secret", env_var="PYTHONPATH")


def test_command_center_requires_control_header_for_mutation():
    from fastapi.testclient import TestClient
    from command_center.web import app

    with TestClient(app) as client:
        response = client.post("/api/datasets", json={"name": "forged"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "CONTROL_HEADER_REQUIRED"


def test_command_center_rejects_cross_origin_mutation():
    from fastapi.testclient import TestClient
    from command_center.web import app

    with TestClient(app) as client:
        response = client.post(
            "/api/datasets",
            json={"name": "forged"},
            headers={"x-m2s-command-center": "1", "origin": "https://evil.example"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "ORIGIN_REJECTED"


def test_command_center_errors_do_not_echo_exception_details(monkeypatch):
    from fastapi.testclient import TestClient
    import command_center.web as web

    def exploding(*_args, **_kwargs):
        raise ValueError("/secret/internal/path must not leak")

    monkeypatch.setattr(web, "ingest", exploding)
    with TestClient(web.app) as client:
        response = client.post(
            "/api/datasets/1/ingest",
            json={"path": "/tmp/x"},
            headers={"x-m2s-command-center": "1"},
        )
        assert response.status_code == 400
        assert "/secret/internal/path" not in response.text
        assert response.json()["detail"]["code"] == "REQUEST_FAILED"
