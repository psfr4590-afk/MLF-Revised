from __future__ import annotations

import json
from pathlib import Path

from pipeline.provenance import config_identities, snapshot_configs


def test_configuration_identities_are_independent():
    cfg = {
        "crawl": {"sources": {"web": True}},
        "clean": {"config_file": "clean.yaml"},
        "dedup": {"threshold": 0.9},
        "weight": {"strategy": "upsample"},
        "tokenizer": {"vocab_size": 512},
        "shard": {"sequence_length": 128},
        "train": {"model_preset": "85M", "lr_max": 1e-3, "total_steps": 1000},
        "export": {"quant": "F16"},
    }
    ids = config_identities(cfg, pipeline_sha256="legacy")
    assert ids["dataset_config_sha256"]
    assert ids["tokenizer_config_sha256"]
    assert ids["shard_config_sha256"]
    assert ids["train_config_sha256"]
    assert ids["model_config_sha256"]
    assert ids["export_config_sha256"]
    assert ids["dataset_config_sha256"] != ids["train_config_sha256"]
    assert ids["train_config_sha256"] != ids["model_config_sha256"]


def test_configuration_snapshots_are_immutable(tmp_path: Path):
    cfg = {"crawl": {}, "clean": {}, "dedup": {}, "weight": {}, "tokenizer": {}, "shard": {}, "train": {}, "export": {}}
    ids = config_identities(cfg, pipeline_sha256="legacy")
    paths = snapshot_configs(tmp_path, cfg, ids)
    original = (tmp_path / paths["dataset"]).read_text(encoding="utf-8")
    cfg["train"]["total_steps"] = 999
    snapshot_configs(tmp_path, cfg, ids)
    assert (tmp_path / paths["dataset"]).read_text(encoding="utf-8") == original
    assert json.loads((tmp_path / paths["dataset"]).read_text(encoding="utf-8"))["crawl"] == {}
