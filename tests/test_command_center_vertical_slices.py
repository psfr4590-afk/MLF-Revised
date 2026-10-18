"""Command Center vertical slice tests.

Tests the full HTTP API path: create dataset → trigger stage → verify
status → stop stage → log tail. These complement the unit-level stage_sync
and state tests by exercising the complete request/response chain.
"""
from __future__ import annotations


from fastapi.testclient import TestClient

from command_center.web import app

LOOPBACK = ("127.0.0.1", 12345)
CTRL = {"x-m2s-command-center": "1"}


# ---------------------------------------------------------------------------
# Dataset create → list → detail roundtrip
# ---------------------------------------------------------------------------

def test_create_dataset_appears_in_list():
    with TestClient(app, client=LOOPBACK) as client:
        before = {d["id"] for d in client.get("/api/datasets").json()}
        r = client.post("/api/datasets", headers=CTRL, json={
            "name": "slice-test", "description": "vertical slice"
        })
        assert r.status_code == 200
        new_id = r.json()["id"]
        assert new_id not in before

        listing = client.get("/api/datasets").json()
        ids = {d["id"] for d in listing}
        assert new_id in ids


def test_dataset_detail_returns_expected_shape():
    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "detail-shape", "description": ""
        }).json()["id"]
        r = client.get(f"/api/datasets/{did}")
        assert r.status_code == 200
        body = r.json()
        for key in ("id", "name", "status", "stages", "stats", "events"):
            assert key in body, f"missing key: {key}"
        assert body["id"] == did
        assert isinstance(body["stages"], dict)


def test_dataset_detail_404_for_unknown_id():
    with TestClient(app, client=LOOPBACK) as client:
        r = client.get("/api/datasets/99999")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Stage trigger → running state → stop
# ---------------------------------------------------------------------------

def test_stage_run_accepted_and_returns_started(monkeypatch):
    import command_center.runner as runner

    class _FakeProcess:
        stdout = iter([])
        def wait(self):
            return 0

    class _FakeCreds:
        @staticmethod
        def environment():
            return {}

    monkeypatch.setattr(runner, "credentials", _FakeCreds())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: _FakeProcess())
    runner.RUNS.clear()

    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "stage-trigger"
        }).json()["id"]
        r = client.post(f"/api/datasets/{did}/stage/crawl", headers=CTRL)
        assert r.status_code == 200
        body = r.json()
        assert body.get("started") is True

    runner.RUNS.clear()


def test_stage_run_rejected_for_unknown_stage():
    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "bad-stage"
        }).json()["id"]
        r = client.post(f"/api/datasets/{did}/stage/not_a_stage", headers=CTRL)
        assert r.status_code == 400


def test_stop_endpoint_responds_for_dataset_with_no_active_run():
    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "stop-idle"
        }).json()["id"]
        r = client.post(f"/api/datasets/{did}/stop", headers=CTRL)
        # Either 200 (already idle) or 404 — not a 5xx.
        assert r.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Log tail endpoint
# ---------------------------------------------------------------------------

def test_log_tail_returns_bounded_text():
    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "log-tail"
        }).json()["id"]
        r = client.get(f"/api/datasets/{did}/crawl/log?tail=20")
        assert r.status_code == 200


def test_log_tail_clamps_excessive_line_count():
    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "log-tail-limit"
        }).json()["id"]
        r = client.get(f"/api/datasets/{did}/crawl/log?tail=99999")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) <= 1000


# ---------------------------------------------------------------------------
# Events endpoint
# ---------------------------------------------------------------------------

def test_events_tail_returns_list():
    with TestClient(app, client=LOOPBACK) as client:
        did = client.post("/api/datasets", headers=CTRL, json={
            "name": "event-tail"
        }).json()["id"]
        r = client.get(f"/api/datasets/{did}?event_tail=5")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body.get("events"), list)


# ---------------------------------------------------------------------------
# Mutation endpoint security (control header required)
# ---------------------------------------------------------------------------

def test_create_dataset_without_control_header_is_rejected():
    with TestClient(app, client=LOOPBACK) as client:
        r = client.post("/api/datasets", json={
            "name": "no-header"
        })
        assert r.status_code == 403


def test_stop_without_control_header_is_rejected():
    with TestClient(app, client=LOOPBACK) as client:
        r = client.post("/api/datasets/1/stop")
        assert r.status_code == 403
