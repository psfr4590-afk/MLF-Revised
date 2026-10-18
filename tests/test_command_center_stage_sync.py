import time


def test_refresh_stats_counts_crawl_documents_before_clean(tmp_path, monkeypatch):
    import command_center.store as store_module
    from pipeline.integrity import write_manifest

    monkeypatch.setattr(store_module, "DATASETS", tmp_path)
    store = store_module.DatasetStore()
    meta = store.create("crawl-stats", group_config={"id": "test", "name": "test", "sources": {}})
    root = store.path(meta["id"])
    artifact = root / "scratch" / "01_crawled.jsonl"
    artifact.write_text(
        '{"text":"one two three"}\n{"text":"four five"}\n',
        encoding="utf-8",
    )
    write_manifest(artifact, kind="crawl", rows=2)

    state = store.refresh_pipeline_state(meta["id"])
    assert state["stats"]["documents"] == 2
    assert state["stats"]["words"] == 5
    assert state["stages"]["crawl"] == "complete"


def test_stage_lifecycle_records_start_and_completion_events(monkeypatch, tmp_path):
    import command_center.runner as runner
    import command_center.store as store_module
    from pipeline.integrity import write_manifest

    monkeypatch.setattr(store_module, "DATASETS", tmp_path)
    test_store = store_module.DatasetStore()
    meta = test_store.create("event-state", group_config={"id": "test", "name": "test", "sources": {}})
    root = test_store.path(meta["id"])
    artifact = root / "scratch" / "01_crawled.jsonl"
    artifact.write_text('{"text":"one two"}\n', encoding="utf-8")
    write_manifest(artifact, kind="crawl", rows=1)

    monkeypatch.setattr(runner, "store", test_store)
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    class Credentials:
        @staticmethod
        def environment():
            return {}

    class Process:
        stdout = ()

        def wait(self):
            return 0

    monkeypatch.setattr(runner, "credentials", Credentials())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Process())
    runner.RUNS.clear()
    assert runner.start_stage(meta["id"], "crawl")["started"] is True

    deadline = time.time() + 5
    while time.time() < deadline:
        events = test_store.tail_events(meta["id"], limit=10)
        names = [event["event"] for event in events]
        if "stage.started" in names and "stage.completed" in names:
            break
        time.sleep(0.01)

    events = test_store.tail_events(meta["id"], limit=10)
    names = [event["event"] for event in events]
    assert "stage.started" in names
    assert "stage.completed" in names
    assert test_store.get(meta["id"])["stats"]["documents"] == 1
    runner.RUNS.clear()
