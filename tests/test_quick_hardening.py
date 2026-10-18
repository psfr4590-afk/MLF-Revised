from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from command_center.config import load_pipeline_config
from command_center.secrets import _key
from pipeline.orchestrator import _source_definition_paths, _write_source_manifest
from pipeline.shardwriter.shard_writer import ShardDataLoader

ROOT = Path(__file__).resolve().parents[1]


def _write_shard_fixture(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    path = shard_dir / "shard_00000_train.bin"
    path.write_bytes(np.arange(64, dtype=np.uint16).tobytes())
    manifest = {
        "version": 1,
        "dtype": "uint16",
        "sequence_length": 8,
        "vocab_size": 64,
        "files": [{"name": path.name, "size": path.stat().st_size, "sha256": ShardDataLoader._compute_sha256(path)}],
    }
    (shard_dir / "shards.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return shard_dir


def test_shard_loader_rejects_cursor_beyond_shard_bounds(tmp_path):
    shard_dir = _write_shard_fixture(tmp_path)
    loader = ShardDataLoader(shard_dir, "train", 8)
    state = loader.state_dict()
    state["pos"] = 65
    with pytest.raises(RuntimeError, match="exceeds shard bounds"):
        loader.load_state_dict(state)


def test_command_center_uses_canonical_host_and_port():
    config = load_pipeline_config()
    assert config["command_center"]["host"] == "127.0.0.1"
    assert int(config["command_center"]["port"]) == 8000


def test_source_manifest_has_required_source_metadata(tmp_path):
    config = yaml.safe_load((ROOT / "config" / "pipeline_config.yaml").read_text(encoding="utf-8"))
    paths = _source_definition_paths(config, ROOT)
    target = tmp_path / "source_manifest.json"
    _write_source_manifest(target, config, paths)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["retrieval_started_at"]
    assert data["source_definition_files"]
    assert "pipeline_config" not in data["source_definition_files"]
    for source in data["sources"]:
        assert {"kind", "identifier", "revision", "license", "raw_source_sha256"} <= set(source)


def test_credential_store_key_error_is_platform_neutral(monkeypatch):
    monkeypatch.delenv("PIPELINE_CREDENTIAL_KEY", raising=False)
    with pytest.raises(RuntimeError, match="required to access the credential store"):
        _key()
