from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline.trainer.model import ModelConfig
from pipeline.trainer.train import _provenance
from scripts.export_gguf import _enforce_training_provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(path: Path) -> None:
    path.write_text(json.dumps({"schema": 2, "dataset_group": "fixture", "retrieval_started_at": "t", "retrieval_completed_at": "t", "source_definition_files": {}, "sources": [{"kind": "local", "identifier": "fixture", "revision": "embedded", "license": "test", "raw_source_sha256": None, "dataset_group": "fixture"}], "rights_note": "test"}), encoding="utf-8")


def test_training_provenance_requires_shard_manifest(tmp_path):
    output = tmp_path / "output"; output.mkdir(); _source_manifest(tmp_path / "source_manifest.json")
    shards = output / "shards"; shards.mkdir()
    cfg = {"_pipeline_config_sha256": "a" * 64, "pipeline": {"output_dir": str(output)}}
    with pytest.raises(RuntimeError, match="shard manifest"):
        _provenance(cfg, ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=8), shards)


def test_training_provenance_records_source_manifest_hash(tmp_path):
    output = tmp_path / "output"; output.mkdir(); source = tmp_path / "source_manifest.json"; _source_manifest(source)
    shards = output / "shards"; shards.mkdir(); (shards / "shards.manifest.json").write_text("{}", encoding="utf-8")
    cfg = {"_pipeline_config_sha256": "a" * 64, "pipeline": {"output_dir": str(output)}}
    provenance = _provenance(cfg, ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=8), shards)
    assert provenance["schema"] == 3
    assert provenance["config_identities"]["train_config_sha256"]
    assert provenance["config_identities"]["model_config_sha256"] == provenance["model_config_sha256"]
    assert provenance["source_manifest_sha256"] == _sha256(source)


def test_export_provenance_rejects_source_manifest_replacement(tmp_path):
    output = tmp_path / "output"; (output / "shards").mkdir(parents=True); (output / "tokenizer").mkdir(parents=True); (tmp_path / "scratch").mkdir(); source = tmp_path / "source_manifest.json"; _source_manifest(source)
    for path, payload in ((output / "shards" / "shards.manifest.json", {"provenance": {"pipeline_config_sha256": "a"}}), (output / "tokenizer" / "tokenizer.json.manifest.json", {"provenance": {"pipeline_config_sha256": "a"}}), (tmp_path / "scratch" / "04_weighted.jsonl.manifest.json", {"provenance": {"pipeline_config_sha256": "a"}})):
        path.write_text(json.dumps(payload), encoding="utf-8")
    provenance = {
        "schema": 3,
        "run_id": "fixture",
        "config_identities": {
            "dataset_config_sha256": "d",
            "tokenizer_config_sha256": "t",
            "shard_config_sha256": "s",
            "source_definition_sha256": "x",
        },
        "train_config_sha256": "b",
        "model_config_sha256": "c",
        "shard_manifest_sha256": _sha256(output / "shards" / "shards.manifest.json"),
        "source_manifest_sha256": "wrong",
        "seed": 42,
        "parent_artifact_ids": [],
        "tokenizer_artifact_id": "fixture",
        "dataset_artifact_id": "fixture",
        "source_artifact_id": "fixture",
    }
    with pytest.raises(RuntimeError, match="source-manifest identity"):
        _enforce_training_provenance(output, {"provenance": provenance, "model_cfg": {"vocab_size": 8}})


def test_export_accepts_independent_stage_config_identities_without_pipeline_hash_equality(tmp_path):
    import hashlib

    output = tmp_path / "output"
    (output / "shards").mkdir(parents=True)
    (output / "tokenizer").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = tmp_path / "source_manifest.json"
    tokenizer = output / "tokenizer" / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(8)}}}), encoding="utf-8")

    identities = {
        "dataset_config_sha256": "d" * 64,
        "tokenizer_config_sha256": "t" * 64,
        "shard_config_sha256": "s" * 64,
        "source_definition_sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    source.write_text(json.dumps({
        "schema": 2,
        "run_id": "fixture",
        "dataset_group": "fixture",
        "retrieval_started_at": "t",
        "retrieval_completed_at": "t",
        "source_definition_files": {},
        "sources": [{"kind": "local", "identifier": "fixture", "revision": "embedded", "license": "test", "raw_source_sha256": None, "dataset_group": "fixture"}],
        "source_definition_sha256": identities["source_definition_sha256"],
        "rights_note": "test",
    }), encoding="utf-8")

    def manifest(path: Path, provenance: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        path.with_name(path.name + ".manifest.json").write_text(json.dumps({
            "schema": 2, "kind": "fixture", "path": str(path), "size": path.stat().st_size,
            "sha256": _sha256(path), "provenance": provenance,
        }), encoding="utf-8")

    manifest(scratch / "04_weighted.jsonl", {
        "run_id": "fixture", "config_identities": identities | {"pipeline_config_sha256": "historical-" + "a" * 10},
    })
    manifest(output / "tokenizer" / "tokenizer.json", {
        "run_id": "fixture", "config_identities": identities | {"pipeline_config_sha256": "historical-" + "b" * 10},
    })
    tokenizer_manifest = output / "tokenizer" / "tokenizer.json.manifest.json"
    tokenizer_manifest_data = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
    tokenizer_manifest_data["vocab_size"] = 8
    tokenizer_manifest.write_text(json.dumps(tokenizer_manifest_data), encoding="utf-8")
    shard_manifest = output / "shards" / "shards.manifest.json"
    shard_manifest.write_text(json.dumps({
        "schema": 4,
        "files": [],
        "provenance": {"config_identities": identities | {"pipeline_config_sha256": "historical-" + "c" * 10},
            "run_id": "fixture",
            "tokenizer_sha256": _sha256(tokenizer),
        },
    }), encoding="utf-8")

    payload = {
        "provenance": {
            "schema": 3,
            "run_id": "fixture",
            "config_identities": identities,
            "train_config_sha256": "b" * 64,
            "model_config_sha256": "c" * 64,
            "shard_manifest_sha256": _sha256(shard_manifest),
            "source_manifest_sha256": _sha256(source),
            "seed": 42,
            "parent_artifact_ids": [
                f"shard-manifest:{_sha256(shard_manifest)}",
                f"tokenizer-manifest:{_sha256(tokenizer_manifest)}",
                f"dataset-manifest:{_sha256(scratch / '04_weighted.jsonl.manifest.json')}",
                f"source-manifest:{_sha256(source)}",
            ],
            "tokenizer_artifact_id": f"tokenizer-manifest:{_sha256(tokenizer_manifest)}",
            "dataset_artifact_id": f"dataset-manifest:{_sha256(scratch / '04_weighted.jsonl.manifest.json')}",
            "source_artifact_id": f"source-manifest:{_sha256(source)}",
        },
        "model_cfg": {"vocab_size": 8},
    }
    _enforce_training_provenance(output, payload)


@pytest.mark.parametrize("schema", [2, 3])
def test_lineage_validator_accepts_supported_source_manifest_schemas(tmp_path, schema):
    import hashlib
    from scripts.validate_lineage import validate

    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    (output / "tokenizer").mkdir(parents=True)
    (output / "shards").mkdir(parents=True)
    scratch.mkdir()
    weighted = scratch / "04_weighted.jsonl"
    weighted.write_text('{"text":"fixture"}\n', encoding="utf-8")
    (scratch / "04_weighted.jsonl.manifest.json").write_text(json.dumps({"schema": 2, "sha256": hashlib.sha256(weighted.read_bytes()).hexdigest(), "provenance": {}}), encoding="utf-8")
    tokenizer = output / "tokenizer" / "tokenizer.json"
    tokenizer.write_text('{"model":{"vocab":{}}}\n', encoding="utf-8")
    (output / "tokenizer" / "tokenizer.json.manifest.json").write_text(json.dumps({"schema": 2, "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(), "provenance": {}}), encoding="utf-8")
    (output / "shards" / "shards.manifest.json").write_text(json.dumps({"schema": 4, "files": [], "provenance": {}}), encoding="utf-8")
    source = tmp_path / "source_manifest.json"
    source.write_text(json.dumps({"schema": schema, "run_id": "fixture-run", "source_definition_sha256": hashlib.sha256(b"{}").hexdigest(), "source_definition_files": {}, "sources": []}), encoding="utf-8")

    checks, code = validate(output)

    assert code == 0
    assert not any(item["name"] == "Configuration" and item["status"] == "FAIL" for item in checks)
    assert next(item for item in checks if item["name"] == "Source configuration")["status"] == "PASS"


def test_lineage_validator_rejects_unknown_source_manifest_schema(tmp_path):
    import hashlib
    from scripts.validate_lineage import validate

    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    (output / "tokenizer").mkdir(parents=True)
    (output / "shards").mkdir(parents=True)
    scratch.mkdir()
    weighted = scratch / "04_weighted.jsonl"
    weighted.write_text('{"text":"fixture"}\n', encoding="utf-8")
    (scratch / "04_weighted.jsonl.manifest.json").write_text(json.dumps({"schema": 2, "sha256": hashlib.sha256(weighted.read_bytes()).hexdigest(), "provenance": {}}), encoding="utf-8")
    tokenizer = output / "tokenizer" / "tokenizer.json"
    tokenizer.write_text('{"model":{"vocab":{}}}\n', encoding="utf-8")
    (output / "tokenizer" / "tokenizer.json.manifest.json").write_text(json.dumps({"schema": 2, "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(), "provenance": {}}), encoding="utf-8")
    (output / "shards" / "shards.manifest.json").write_text(json.dumps({"schema": 4, "files": [], "provenance": {}}), encoding="utf-8")
    source = tmp_path / "source_manifest.json"
    source.write_text(json.dumps({"schema": 4, "run_id": "fixture-run", "source_definition_sha256": hashlib.sha256(b"{}").hexdigest(), "source_definition_files": {}, "sources": []}), encoding="utf-8")

    checks, code = validate(output)

    assert code == 2
    configuration = next(item for item in checks if item["name"] == "Configuration")
    assert configuration["status"] == "FAIL"
    assert "unsupported schema" in configuration["detail"]
