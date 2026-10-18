"""Tests for orchestrator critical paths.

Covers: Pipeline init, _should_skip, stage_clean, stage_weight,
_source_manifest_is_valid, and Pipeline.run() failure arc.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipeline.integrity import sha256_file, write_manifest
from pipeline.orchestrator import (
    Pipeline,
    _source_manifest_is_valid,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _write_smoke_config(root: Path, out_dir: Path, scratch_dir: Path) -> Path:
    """Write a minimal but fully valid pipeline config to root/pipeline_config.yaml."""
    cfg = {
        "pipeline": {
            "name": "test-pipeline",
            "output_dir": str(out_dir),
            "scratch_dir": str(scratch_dir),
            "resume": False,
        },
        "stages": {
            "crawl": False, "clean": False, "dedup": False, "weight": False,
            "tokenize": False, "shard": False, "train": False, "export": False,
        },
        "crawl": {
            "dataset_groups_file": "config/dataset_groups.yaml",
            "source_weights_file": "config/source_weights.yaml",
            "sources": {"web": False, "github": False, "arxiv": False,
                        "huggingface": False, "google": False},
        },
        "clean": {"config_file": "config/cleaner_config.yaml"},
        "embed_dedup": {"mode": "auto", "model": "all-MiniLM-L6-v2"},
        "weight": {"strategy": "upsample"},
        "tokenizer": {"vocab_size": 512, "output_path": str(out_dir / "tokenizer")},
        "shard": {
            "sequence_length": 128, "shard_size_tokens": 129, "val_fraction": 0.1, "dtype": "uint16",
            "output_dir": str(out_dir / "shards"),
        },
        "train": {"resume": False, "allow_cpu_training": True, "vocab_size": 512, "seq_len": 128, "total_steps": 1, "batch_size": 1, "grad_accum_steps": 1, "eval_every_steps": 1, "eval_batches": 1, "checkpoint_every_steps": 1, "keep_checkpoints": 1},
        "export": {"format": "gguf", "llamacpp_dir": "third_party/llama.cpp",
                   "quant": "Q4_K_M", "model_name": "test"},
    }
    path = root / "pipeline_config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def pipeline_env(tmp_path):
    """Return (Pipeline, out_dir, scratch_dir) with a minimal valid config."""
    out_dir = tmp_path / "output"
    scratch_dir = tmp_path / "scratch"
    out_dir.mkdir(parents=True)
    scratch_dir.mkdir(parents=True)
    cfg_path = _write_smoke_config(tmp_path, out_dir, scratch_dir)
    pl = Pipeline(str(cfg_path))
    return pl, out_dir, scratch_dir


def _write_crawled_jsonl(path: Path, n: int = 3) -> Path:
    """Write a valid 01_crawled.jsonl with n documents and its manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "doc_id": f"doc-{i}",
            "url": f"https://example.com/{i}",
            "text": f"document {i} has enough words and substantially more than two hundred characters to pass the production cleaner length gate. This fixture intentionally contains realistic text so the orchestrator test exercises the cleaning stage rather than accidentally producing an empty artifact.",
            "source": "web",
            "domain": "example.com",
            "quality_score": 1.0,
        })
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(path, kind="crawl", rows=n)
    return path


# ---------------------------------------------------------------------------
# _source_manifest_is_valid
# ---------------------------------------------------------------------------

def _valid_manifest(dataset_group: str = "swe_cs_systems") -> dict:
    return {
        "schema": 2,
        "run_id": "run-abc123",
        "dataset_group": dataset_group,
        "retrieval_started_at": "2026-01-01T00:00:00+00:00",
        "retrieval_completed_at": "2026-01-01T01:00:00+00:00",
        "source_definition_sha256": "a" * 64,
        "source_definition_files": {
            "dataset_groups": {
                "path": "config/dataset_groups.yaml",
                "sha256": "b" * 64,
            }
        },
        "sources": [
            {
                "kind": "web",
                "identifier": "https://example.com",
                "revision": None,
                "license": "unknown",
                "raw_source_sha256": None,
                "dataset_group": dataset_group,
            }
        ],
        "rights_note": "License must be verified.",
    }


def test_source_manifest_is_valid_accepts_well_formed_manifest(tmp_path):
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    assert _source_manifest_is_valid(path)


def test_source_manifest_is_valid_rejects_missing_file(tmp_path):
    assert not _source_manifest_is_valid(tmp_path / "nonexistent.json")


def test_source_manifest_is_valid_rejects_missing_required_field(tmp_path):
    manifest = _valid_manifest()
    del manifest["retrieval_completed_at"]
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _source_manifest_is_valid(path)


def test_source_manifest_is_valid_rejects_old_schema(tmp_path):
    manifest = _valid_manifest()
    manifest["schema"] = 1
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _source_manifest_is_valid(path)


def test_source_manifest_is_valid_rejects_wrong_group(tmp_path):
    manifest = _valid_manifest(dataset_group="group-a")
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _source_manifest_is_valid(path, expected_group_id="group-b")


def test_source_manifest_is_valid_rejects_malformed_source_sha256(tmp_path):
    manifest = _valid_manifest()
    manifest["sources"][0]["raw_source_sha256"] = "not-a-hex-digest"
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _source_manifest_is_valid(path)


def test_source_manifest_is_valid_rejects_empty_sources_list(tmp_path):
    manifest = _valid_manifest()
    manifest["sources"] = []
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _source_manifest_is_valid(path)


# ---------------------------------------------------------------------------
# Pipeline.__init__
# ---------------------------------------------------------------------------

def test_pipeline_init_sets_expected_attributes(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    assert pl._out == out_dir.resolve()
    assert pl._scratch == scratch_dir.resolve()
    assert pl.dataset_id is None
    assert pl._resume is False
    assert pl.cfg["_run_id"]


def test_pipeline_init_creates_output_dirs(tmp_path):
    out_dir = tmp_path / "deep" / "output"
    scratch_dir = tmp_path / "deep" / "scratch"
    cfg_path = _write_smoke_config(tmp_path, out_dir, scratch_dir)
    Pipeline(str(cfg_path))
    assert out_dir.is_dir()
    assert scratch_dir.is_dir()


def test_pipeline_init_rejects_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        Pipeline(str(tmp_path / "does_not_exist.yaml"))


# ---------------------------------------------------------------------------
# _should_skip
# ---------------------------------------------------------------------------

def test_should_skip_returns_false_when_artifact_absent(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    absent = scratch_dir / "01_crawled.jsonl"
    assert not pl._should_skip(absent, "crawl", {})


def test_should_skip_returns_false_when_resume_disabled(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    path = scratch_dir / "02_cleaned.jsonl"
    path.write_text('{"text":"something"}\n', encoding="utf-8")
    write_manifest(path, kind="clean", rows=1)
    pl._resume = False
    assert not pl._should_skip(path, "clean", {})


def test_should_skip_returns_true_for_valid_artifact_with_resume(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    path = scratch_dir / "02_cleaned.jsonl"
    path.write_text('{"text":"something"}\n', encoding="utf-8")
    write_manifest(path, kind="clean", rows=1)
    pl._resume = True
    # No expected provenance — artifact_valid passes when manifest is correct.
    assert pl._should_skip(path, "clean", {})


# ---------------------------------------------------------------------------
# stage_clean
# ---------------------------------------------------------------------------

def test_stage_clean_writes_output_and_manifest(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    in_path = _write_crawled_jsonl(scratch_dir / "01_crawled.jsonl")
    out_path = pl.stage_clean(in_path)
    assert out_path.is_file()
    assert out_path.with_name(out_path.name + ".manifest.json").is_file()


def test_stage_clean_drops_short_documents(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    in_path = scratch_dir / "01_crawled.jsonl"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    # One long doc (should pass), one too short (should drop).
    docs = [
        json.dumps({
            "doc_id": "keep",
            "url": "https://example.com/1",
            "text": "this document has enough words to survive the production cleaner gate comfortably. It is deliberately longer than two hundred characters so this test verifies that a valid document is retained while the deliberately short document below is rejected by the configured minimum length gate.",
            "source": "web",
        }),
        json.dumps({
            "doc_id": "drop",
            "url": "https://example.com/2",
            "text": "too short",
            "source": "web",
        }),
    ]
    in_path.write_text("\n".join(docs) + "\n", encoding="utf-8")
    write_manifest(in_path, kind="crawl", rows=2)
    out_path = pl.stage_clean(in_path)
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) < 2
    assert all(r.get("doc_id") != "drop" for r in records)


def test_stage_clean_skips_when_output_valid_and_resume_true(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    in_path = _write_crawled_jsonl(scratch_dir / "01_crawled.jsonl")
    out_path = scratch_dir / "02_cleaned.jsonl"

    # Build a genuinely valid cached artifact using the same provenance
    # contract stage_clean() uses when deciding whether resume is safe.
    provenance = pl._provenance(
        "clean",
        in_path,
        {"clean_config_sha256": sha256_file(pl._clean_config_path)},
    )
    out_path.write_text(
        '{"doc_id":"cached","url":"https://example.com/cached",'
        '"source":"web","text":"cached artifact with enough content to satisfy '
        'the cleaner contract and remain a valid resumable output.",'
        '"clean_action":"kept","clean_score":0.0}\n',
        encoding="utf-8",
    )
    write_manifest(
        out_path,
        kind="clean",
        rows=1,
        provenance=provenance,
    )

    pl._resume = True
    result = pl.stage_clean(in_path, out_path)

    assert result == out_path
    # Content should still be the cached version, not re-cleaned.
    assert "cached artifact" in out_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# stage_weight
# ---------------------------------------------------------------------------

def test_stage_weight_writes_output_and_manifest(pipeline_env):
    pl, out_dir, scratch_dir = pipeline_env
    in_path = scratch_dir / "03_deduped.jsonl"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    in_path.write_text(
        json.dumps({
            "doc_id": "w1",
            "url": "https://github.com/example/repo",
            "text": "code example function main",
            "source": "github",
            "domain": "github.com",
            "quality_score": 0.9,
        }) + "\n",
        encoding="utf-8",
    )
    write_manifest(in_path, kind="dedup", rows=1)
    out_path = pl.stage_weight(in_path)
    assert out_path.is_file()
    assert out_path.with_name(out_path.name + ".manifest.json").is_file()
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert all("final_weight" in r for r in records)


# ---------------------------------------------------------------------------
# Pipeline.run() failure arc
# ---------------------------------------------------------------------------

def test_pipeline_run_records_failed_status_when_stage_raises(tmp_path, monkeypatch):
    out_dir = tmp_path / "output"
    scratch_dir = tmp_path / "scratch"
    out_dir.mkdir(parents=True)
    scratch_dir.mkdir(parents=True)

    # Enable just the crawl stage so we can intercept it.
    cfg_path = _write_smoke_config(tmp_path, out_dir, scratch_dir)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["stages"]["crawl"] = True
    cfg_path.write_text(yaml.dump(data), encoding="utf-8")

    pl = Pipeline(str(cfg_path))

    monkeypatch.setattr(
        pl, "stage_crawl",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("injected failure")),
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        pl.run()

    run_id = pl.cfg["_run_id"]
    manifest_path = out_dir / "runs" / run_id / "run_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("final_status") == "FAILED"

def test_offline_crawl_clean_dedup_weight_vertical_slice(
    pipeline_env, monkeypatch
):
    """Exercise crawl -> clean -> dedup -> weight without network access."""
    import pipeline.orchestrator as orchestrator
    from pipeline.types import Document

    pl, out_dir, scratch_dir = pipeline_env
    pl._resume = False

    # Enable one synthetic seed so stage_crawl invokes the fake crawler.
    crawl_cfg = pl.cfg.setdefault("crawl", {})
    web_cfg = dict(crawl_cfg.get("web", {}))
    web_cfg["seed_urls"] = ["https://example.invalid/offline-fixture"]
    crawl_cfg["web"] = web_cfg

    monkeypatch.setattr(
        pl,
        "_load_dataset_groups",
        lambda: [{
            "id": "swe_cs_systems",
            "name": "offline integration fixture",
            "sources": {
                "web": True,
                "github": False,
                "arxiv": False,
                "huggingface": False,
                "google": False,
            },
        }],
    )

    base_text = (
        "This deterministic integration fixture describes a software system, "
        "its configuration, validation rules, data processing, and test results. "
        "It contains enough ordinary technical prose to pass the production "
        "cleaner's minimum-length requirement without triggering refusal filters. "
        "The pipeline should preserve this content and its artifact provenance."
    )
    alternate_text = base_text.replace("deterministic", "Deterministic", 1)

    class FakeCrawler:
        def __init__(self, config, weights, signals):
            self.stats = {"errors": 0, "skipped": 0}

        def crawl(self):
            yield Document(
                doc_id="offline-doc-1",
                url="https://example.invalid/fixture/1",
                source="web",
                title="Offline fixture one",
                text=base_text,
                meta={},
            )
            yield Document(
                doc_id="offline-doc-2",
                url="https://example.invalid/fixture/2",
                source="web",
                title="Offline fixture two",
                text=alternate_text,
                meta={},
            )

        def observability(self):
            return {
                "request_count": 1,
                "document_count": 2,
                "rate_limited": 0,
                "revisions": [],
                "licenses": [],
                "source_hash": "offline-test-source",
            }

    class PassThroughDeduplicator:
        def __init__(self, config):
            self.stats = {"dropped": 0}

        def stream(self, docs, buffer_size=10000):
            yield from docs

    original_load_class = orchestrator._load_class

    def offline_load_class(module, name):
        if module == "pipeline.crawler.web_crawler":
            return FakeCrawler
        if module == "pipeline.embedder.semantic_dedup":
            return PassThroughDeduplicator
        return original_load_class(module, name)

    monkeypatch.setattr(orchestrator, "_load_class", offline_load_class)

    crawled = pl.stage_crawl()
    assert crawled.is_file()
    assert crawled.with_name(crawled.name + ".manifest.json").is_file()

    def read_records(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    crawled_records = read_records(crawled)
    assert len(crawled_records) == 2

    cleaned = pl.stage_clean(crawled)
    assert cleaned.is_file()
    assert cleaned.with_name(cleaned.name + ".manifest.json").is_file()

    cleaned_records = read_records(cleaned)
    assert len(cleaned_records) == 2

    deduped = pl.stage_embed_dedup(cleaned)
    assert deduped.is_file()
    assert deduped.with_name(deduped.name + ".manifest.json").is_file()

    deduped_records = read_records(deduped)
    assert len(deduped_records) == 1

    weighted = pl.stage_weight(deduped)
    assert weighted.is_file()
    assert weighted.with_name(weighted.name + ".manifest.json").is_file()

    weighted_records = read_records(weighted)
    assert weighted_records
    assert all("final_weight" in record for record in weighted_records)

    for artifact in (crawled, cleaned, deduped, weighted):
        manifest_path = artifact.with_name(artifact.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("rows", 0) >= 1
