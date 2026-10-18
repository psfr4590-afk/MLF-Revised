from __future__ import annotations

from pathlib import Path

import pytest

import command_center.service as service


def test_service_delegates_dataset_operations(monkeypatch):
    calls = []
    monkeypatch.setattr(service.store, "ensure_seed_datasets", lambda: calls.append(("init",)) or "seeded")
    monkeypatch.setattr(service.store, "create", lambda *args, **kwargs: calls.append(("create", args, kwargs)) or "created")
    monkeypatch.setattr(service.store, "ingest_path", lambda *args: calls.append(("ingest", args)) or "ingested")
    monkeypatch.setattr(service, "validate_import_source", lambda path: Path(path))

    assert service.init() == "seeded"
    assert service.add("demo", "desc", "group") == "created"
    assert service.ingest(7, "input.jsonl") == "ingested"
    assert calls[1] == ("create", ("demo", "desc"), {"group_id": "group"})
    assert calls[2] == ("ingest", (7, Path("input.jsonl")))


def test_service_stage_validates_and_delegates(monkeypatch):
    monkeypatch.setattr(service, "start_stage", lambda did, name: (did, name))
    assert service.stage(4, "clean") == (4, "clean")
    with pytest.raises(ValueError, match="Unknown stage: nope"):
        service.stage(4, "nope")


def test_service_status_handles_single_and_all(monkeypatch):
    seen = []
    monkeypatch.setattr(service.store, "refresh_pipeline_state", lambda did: seen.append(did) or {"id": did})
    monkeypatch.setattr(service.store, "list", lambda: [{"id": 1}, {"id": 2}])

    assert service.status(9) == {"id": 9}
    assert service.status() == [{"id": 1}, {"id": 2}]
    assert seen == [9, 1, 2]


def test_service_delegates_groups_and_credentials(monkeypatch):
    monkeypatch.setattr(service, "load_groups", lambda: ["group"])
    monkeypatch.setattr(service.credentials, "list", lambda: ["credential"])
    monkeypatch.setattr(service.credentials, "set", lambda *args: args)
    monkeypatch.setattr(service.credentials, "delete", lambda name: ("delete", name))
    monkeypatch.setattr(service.credentials, "test", lambda name: ("test", name))

    assert service.groups() == ["group"]
    assert service.credential_list() == ["credential"]
    assert service.credential_set("name", "secret", "provider", "kind", "ENV", "desc", "identity") == (
        "name", "secret", "provider", "kind", "ENV", "desc", "identity"
    )
    assert service.credential_delete("name") == ("delete", "name")
    assert service.credential_test("name") == ("test", "name")


def test_service_delegates_crawl_queries(monkeypatch):
    monkeypatch.setattr(service.store, "crawl_stats", lambda did: ("stats", did))
    monkeypatch.setattr(service.store, "crawl_domains", lambda did: ("domains", did))
    monkeypatch.setattr(service.store, "crawl_log", lambda did, tail: ("log", did, tail))

    assert service.crawl_stats(3) == ("stats", 3)
    assert service.crawl_domains(3) == ("domains", 3)
    assert service.crawl_log(3, 12) == ("log", 3, 12)


def test_service_rejects_ingest_outside_managed_import_root():
    with pytest.raises(ValueError, match="inside the configured imports directory"):
        service.ingest(7, Path("/etc/passwd"))
