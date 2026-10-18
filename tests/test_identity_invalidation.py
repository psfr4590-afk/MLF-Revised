from __future__ import annotations

import copy
import json

from pipeline.integrity import artifact_valid, write_manifest
from pipeline.provenance import config_identities, stable_hash


def _config():
    return {
        "crawl": {"sources": {"web": True}, "seed_urls": ["https://example.test/a"]},
        "clean": {"min_chars": 20},
        "dedup": {"threshold": 0.92},
        "weight": {"default_weight": 1.0},
        "tokenizer": {"vocab_size": 4096, "min_frequency": 2},
        "shard": {"seq_len": 512},
        "train": {"model_preset": "small", "seq_len": 512, "dropout": 0.1, "seed": 7},
        "export": {"quant": "Q4_K_M"},
    }


def test_dataset_identity_changes_when_dataset_semantics_change():
    cfg = _config()
    first = config_identities(cfg, source_definition_hashes={"groups": "a" * 64})
    changed = copy.deepcopy(cfg)
    changed["clean"]["min_chars"] = 21
    second = config_identities(changed, source_definition_hashes={"groups": "a" * 64})
    assert first["dataset_config_sha256"] != second["dataset_config_sha256"]


def test_tokenizer_identity_changes_without_changing_dataset_identity():
    cfg = _config()
    first = config_identities(cfg, source_definition_hashes={"groups": "a" * 64})
    changed = copy.deepcopy(cfg)
    changed["tokenizer"]["vocab_size"] = 8192
    second = config_identities(changed, source_definition_hashes={"groups": "a" * 64})
    assert first["dataset_config_sha256"] == second["dataset_config_sha256"]
    assert first["tokenizer_config_sha256"] != second["tokenizer_config_sha256"]


def test_artifact_reuse_requires_exact_identity_bundle(tmp_path):
    artifact = tmp_path / "weighted.jsonl"
    artifact.write_text('{"id":1}\n', encoding="utf-8")
    identities = config_identities(_config(), source_definition_hashes={"groups": "a" * 64})
    provenance = {
        "schema": 3,
        "stage": "weight",
        "run_id": "run-test",
        "config_identities": identities,
        "identity_bundle_sha256": stable_hash(identities),
    }
    write_manifest(artifact, kind="weight", provenance=provenance)
    assert artifact_valid(artifact, expected_provenance=provenance)

    changed = copy.deepcopy(provenance)
    changed["config_identities"] = dict(changed["config_identities"])
    changed["config_identities"]["dataset_config_sha256"] = "b" * 64
    changed["identity_bundle_sha256"] = stable_hash(changed["config_identities"])
    assert not artifact_valid(artifact, expected_provenance=changed)

    manifest_path = tmp_path / "weighted.jsonl.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["provenance"]["identity_bundle_sha256"] == provenance["identity_bundle_sha256"]
    manifest["provenance"]["config_identities"]["dataset_config_sha256"] = "c" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not artifact_valid(artifact)
