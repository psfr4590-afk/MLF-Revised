from __future__ import annotations

import json
from pathlib import Path

from pipeline.integrity import artifact_valid, sha256_file, write_manifest
from pipeline.provenance import config_identities, stable_hash


def test_config_identities_are_stable_and_runtime_paths_are_not_semantic():
    first = {
        "pipeline": {"output_dir": ".acceptance/a", "scratch_dir": ".acceptance/s"},
        "crawl": {"sources": {"local_files": True}},
        "tokenizer": {"type": "bpe", "vocab_size": 512},
        "shard": {"sequence_length": 32},
        "train": {"total_steps": 2, "model_preset": "85M", "resume": False},
        "export": {"format": "gguf"},
    }
    second = json.loads(json.dumps(first))
    second["pipeline"]["output_dir"] = ".acceptance/other-output"
    assert config_identities(first) == config_identities(second)
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_artifact_manifest_hash_and_tamper_detection(tmp_path: Path):
    artifact = tmp_path / "weighted.jsonl"
    artifact.write_text('{"text":"fixture"}\n', encoding="utf-8")
    manifest = write_manifest(artifact, kind="dataset", rows=1, provenance={"run_id": "test-run"})
    assert manifest.is_file()
    assert artifact_valid(artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["sha256"] == sha256_file(artifact)
    artifact.write_text('{"text":"tampered"}\n', encoding="utf-8")
    assert not artifact_valid(artifact)


def test_manifest_tampering_is_detected(tmp_path: Path):
    artifact = tmp_path / "tokenizer.json"
    artifact.write_text('{"model":{"vocab":{}}}', encoding="utf-8")
    manifest = write_manifest(artifact, kind="tokenizer", provenance={"run_id": "test-run"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not artifact_valid(artifact)
