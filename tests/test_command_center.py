from fastapi.testclient import TestClient

from command_center.web import app


LOOPBACK_CLIENT = ("127.0.0.1", 12345)
REMOTE_CLIENT = ("192.0.2.10", 54321)
CONTROL_HEADERS = {"x-m2s-command-center": "1"}


def test_command_center_seeds_and_serves():
    with TestClient(app, client=LOOPBACK_CLIENT) as client:
        response = client.get('/api/datasets')
        assert response.status_code == 200
        datasets = response.json()
        assert len(datasets) >= 4
        assert [d['group_id'] for d in datasets[:4]] == [
            'swe_cs_systems',
            'ai_ml_cybersec_dataeng',
            'sci_reasoning_forensics_formal',
            'domain_finance_bio_robotics',
        ]
        page = client.get('/')
        assert page.status_code == 200
        assert page.encoding == 'utf-8'
        assert 'Model Lab Command Center' in page.text
        assert 'Single local operator interface' in page.text
        assert 'authoritative SQLite state' in page.text
        assert 'M²S Model Training Pipeline' in page.content.decode('utf-8')


def test_command_center_rejects_remote_api_client():
    with TestClient(app, client=REMOTE_CLIENT) as client:
        response = client.get('/api/system')
        assert response.status_code == 403
        assert response.json()["detail"].startswith("Command center is localhost-only.")


def test_credential_store_roundtrip(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    from command_center.secrets import CredentialStore

    monkeypatch.setenv("PIPELINE_CREDENTIAL_KEY", Fernet.generate_key().decode())
    path = tmp_path / "credentials.json"
    store = CredentialStore(path)
    item = store.set("github", "super-secret", "GitHub", "API token", "GITHUB_TOKEN", "test", "test-user")

    assert item["stored"] is True
    assert item["identity"] == "test-user"
    assert store.reveal("github") == "super-secret"
    assert store.environment()["GITHUB_TOKEN"] == "super-secret"
    assert "super-secret" not in path.read_text(encoding="utf-8")


def test_credential_api_does_not_return_secret():
    from command_center.web import app
    from command_center.secrets import credentials
    from fastapi.testclient import TestClient

    original = credentials.path
    try:
        import tempfile
        credentials.path = __import__("pathlib").Path(tempfile.mkdtemp()) / "credentials.json"
        import os
        from cryptography.fernet import Fernet
        os.environ["PIPELINE_CREDENTIAL_KEY"] = Fernet.generate_key().decode()
        with TestClient(app, client=LOOPBACK_CLIENT) as client:
            r = client.post("/api/credentials", headers=CONTROL_HEADERS, json={
                "name": "test", "secret": "do-not-return", "provider": "custom",
                "kind": "token", "env_var": "GITHUB_TOKEN", "identity": "user"
            })
            assert r.status_code == 200
            assert "secret" not in r.json()
            listing = client.get("/api/credentials")
            assert listing.status_code == 200
            body = listing.json()
            assert body[0]["name"] == "test"
            assert "do-not-return" not in listing.text
    finally:
        credentials.path = original


def test_dashboard_api_exposes_persisted_observability():
    with TestClient(app, client=LOOPBACK_CLIENT) as client:
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        body = response.json()
        assert set(["run", "stage", "training", "dataset", "hardware", "provenance", "checkpoints", "artifacts", "warnings", "errors"]) <= set(body)


def test_run_list_and_missing_run_contract():
    with TestClient(app, client=LOOPBACK_CLIENT) as client:
        response = client.get("/api/runs")
        assert response.status_code == 200
        assert "runs" in response.json()
        missing = client.get("/api/runs/does-not-exist")
        assert missing.status_code == 404


def test_command_center_browser_is_operational_not_a_landing_page():
    with TestClient(app, client=LOOPBACK_CLIENT) as client:
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        for marker in [
            'id="metrics"', 'id="run"', 'id="training"', 'id="dataset"',
            'id="hardware"', 'id="credentials"', 'id="provenance"', 'id="stages"', 'id="runs"',
            '/api/dashboard', '/api/system', '/api/runs'
        ]:
            assert marker in html
        assert "Dataset lifecycle" not in html
        assert "Machine status" not in html
        assert "Local Command Center" not in html


def test_command_center_exposes_four_credential_presets_without_secrets():
    with TestClient(app, client=LOOPBACK_CLIENT) as client:
        response = client.get("/api/credentials/presets")
        assert response.status_code == 200
        presets = response.json()
        assert [p["name"] for p in presets] == ["github", "huggingface", "google_api", "google_cx"]
        for preset in presets:
            assert "secret" not in preset
        html = client.get("/").text
        assert 'id="credentials"' in html
        assert {p["env_var"] for p in presets} == {
            "GITHUB_TOKEN", "HF_TOKEN", "GOOGLE_API_KEY", "GOOGLE_CX"
        }


def test_runner_reconciles_dead_worker_state(tmp_path, monkeypatch):
    import command_center.runner as runner

    dataset = {
        "id": 77,
        "status": "RUNNING",
        "stages": {"crawl": "running"},
    }
    state_path = tmp_path / "runner_state.json"
    state_path.write_text(
        '{"schema":1,"dataset_id":77,"stage":"crawl","pid":99999999,"status":"running"}',
        encoding="utf-8",
    )
    updates = []
    events = []

    monkeypatch.setattr(runner.store, "list", lambda: [dataset])
    monkeypatch.setattr(runner.store, "path", lambda did: tmp_path)
    monkeypatch.setattr(runner.store, "update", lambda did, **changes: updates.append((did, changes)))
    monkeypatch.setattr(runner.store, "event", lambda did, event, data=None: events.append((did, event, data)))

    runner.reconcile_orphaned_runs()

    assert updates == [(77, {"stages": {"crawl": "failed"}, "status": "ERROR"})]
    assert events[0][1] == "stage.recovered_orphan"
    assert not state_path.exists()
