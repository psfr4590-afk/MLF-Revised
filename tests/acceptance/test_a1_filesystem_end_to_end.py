from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest

from pipeline.crawler.local_files_crawler import LocalFilesCrawler


class _Weights:
    def domain_weight(self, domain):
        return 1.0, "fixture"

    def content_type_weight(self, content_type):
        return 1.0

    def quality_score(self, text):
        return 1.0

    def code_language_weight(self, language):
        return 1.0


class _Signals:
    def record(self, domain, signal):
        return None


def _deny_network(*args, **kwargs):
    raise AssertionError("Network access attempted during filesystem-only acceptance test")


def test_local_filesystem_ingestion_is_offline_and_hashes_sources(filesystem_corpus, repository_root, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    crawler = LocalFilesCrawler(
        {
            "_project_root": str(repository_root),
            "local_files": {
                "paths": [str(filesystem_corpus)], "recursive": True,
                "extensions": [".md", ".txt"], "max_files": 10, "max_file_bytes": 100_000,
            },
        },
        _Weights(), _Signals(),
    )
    docs = list(crawler.crawl())
    assert len(docs) == 9
    assert {doc.source for doc in docs} == {"local_files"}
    assert all(doc.meta.get("source_mode") == "local" for doc in docs)
    for doc in docs:
        path = Path(doc.meta["local_path"])
        assert path.is_file()
        assert doc.meta["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert doc.text.strip()
    assert crawler.stats["successful_requests"] == 9


def test_fixture_and_pipeline_config_are_explicitly_offline(filesystem_corpus, repository_root):
    import yaml

    assert hashlib.sha256((filesystem_corpus / "alpha.md").read_bytes()).hexdigest() == hashlib.sha256(
        (filesystem_corpus / "alpha_duplicate.md").read_bytes()
    ).hexdigest()
    config = yaml.safe_load((repository_root / "config" / "pipeline_config.a1f.yaml").read_text(encoding="utf-8"))
    groups = yaml.safe_load((repository_root / "config" / "dataset_groups.a1f.yaml").read_text(encoding="utf-8"))
    for source in ("web", "github", "arxiv", "huggingface", "google"):
        assert config["crawl"]["sources"][source] is False
        assert groups["dataset_groups"][0]["sources"][source] is False
    assert config["crawl"]["sources"]["local_files"] is True
    assert config["embed_dedup"]["mode"] == "fallback"
    assert config["embed_dedup"]["allow_model_download"] is False


def test_local_filesystem_ingestion_respects_extension_and_size_limits(filesystem_corpus, repository_root):
    crawler = LocalFilesCrawler(
        {"_project_root": str(repository_root), "local_files": {
            "paths": [str(filesystem_corpus)], "recursive": True,
            "extensions": [".md"], "max_files": 1, "max_file_bytes": 100_000,
        }}, _Weights(), _Signals(),
    )
    docs = list(crawler.crawl())
    assert len(docs) == 1
    assert Path(docs[0].meta["local_path"]).suffix == ".md"
    assert crawler.stats["skipped"] >= 1


@pytest.mark.skipif(os.getenv("MLFRAMEWORK_RUN_FULL_A1F") != "1", reason="set MLFRAMEWORK_RUN_FULL_A1F=1 to run CPU training and GGUF export")
def test_full_a1f_pipeline_has_all_stages_and_export(repository_root, monkeypatch):
    """Run the real pipeline. Strictly requires local llama.cpp converter for GGUF export."""
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    from pipeline.orchestrator import Pipeline

    pipeline = Pipeline(config_path="config/pipeline_config.a1f.yaml", resume=False)
    try:
        pipeline.run(stages="all")
        out = repository_root / ".acceptance" / "output"
        run_manifests = sorted((out / "runs").glob("*/run_manifest.json"))
        assert run_manifests, "Pipeline did not write a run manifest"
        manifest = json.loads(run_manifests[-1].read_text(encoding="utf-8"))
        statuses = {name: data.get("status") for name, data in manifest.get("stages", {}).items()}
        required = ("crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export")
        assert all(statuses.get(stage) in {"PASS", "WARN"} for stage in required), statuses
        source_manifest_path = out.parent / "source_manifest.json"
        assert source_manifest_path.is_file()
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        assert source_manifest.get("sources")
        assert {source.get("kind") for source in source_manifest["sources"]} == {"local_files"}
        assert source_manifest.get("retrieval_started_at") and source_manifest.get("retrieval_completed_at")
        assert (out / "tokenizer" / "tokenizer.json").is_file()
        assert (out / "shards" / "shards.manifest.json").is_file()
        checkpoints = list((out / "checkpoints").glob("*.pt"))
        assert checkpoints, "No model checkpoint found"
        assert all(checkpoint.with_name(checkpoint.name + ".manifest.json").is_file() for checkpoint in checkpoints)
        assert list(run_manifests[-1].parent.glob("dataset_report.json")), "No dataset report/card found"
        gguf_files = list(out.rglob("*.gguf"))
        assert gguf_files, "Export stage did not produce a GGUF file"
    finally:
        pipeline.close()
