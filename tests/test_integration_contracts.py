"""Cross-component contract tests.

Each component here was individually "passing" while the system disagreed with
itself (trainer file names vs exporter/Command Center, hardcoded token IDs,
unverified tensor data). These tests pin the contracts between components using
the trainer's REAL artifact names and manifest schema. They are torch-free.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pipeline.integrity import (
    checkpoint_manifest_valid,
    select_export_checkpoint,
    sha256_file,
)
from pipeline.numerics import NonFiniteLossError, require_finite
from pipeline.tokenizer.special_tokens import special_token_ids


# --------------------------------------------------------------------------- helpers
def _checkpoint(path: Path, payload: bytes = b"x" * 2048, schema: int = 3, kind: str = "checkpoint") -> Path:
    """Write a checkpoint + manifest exactly the way Trainer.save_checkpoint does (schema 3)."""
    path.write_bytes(payload)
    manifest = {"schema": schema, "kind": kind, "size": path.stat().st_size, "sha256": sha256_file(path)}
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- checkpoint contract
def test_exporter_selects_trainer_final_checkpoint_not_latest_periodic(tmp_path):
    for name in ("ckpt_0000001.pt", "ckpt_0000002.pt", "ckpt_best.pt", "ckpt_final.pt"):
        _checkpoint(tmp_path / name)
    assert select_export_checkpoint(tmp_path).name == "ckpt_final.pt"


def test_exporter_falls_back_to_best_then_latest(tmp_path):
    _checkpoint(tmp_path / "ckpt_0000010.pt")
    _checkpoint(tmp_path / "ckpt_0000020.pt")
    assert select_export_checkpoint(tmp_path).name == "ckpt_0000020.pt"
    _checkpoint(tmp_path / "ckpt_best.pt")
    assert select_export_checkpoint(tmp_path).name == "ckpt_best.pt"


def test_exporter_never_selects_tampered_checkpoint(tmp_path):
    final = _checkpoint(tmp_path / "ckpt_final.pt")
    _checkpoint(tmp_path / "ckpt_0000005.pt")
    final.write_bytes(b"y" * 2048)  # tamper after manifest was written
    assert select_export_checkpoint(tmp_path).name == "ckpt_0000005.pt"


def test_exporter_requires_manifest_and_raises_when_nothing_valid(tmp_path):
    (tmp_path / "ckpt_final.pt").write_bytes(b"x" * 2048)  # no manifest
    with pytest.raises(FileNotFoundError):
        select_export_checkpoint(tmp_path)


def test_exporter_accepts_legacy_step_suffixed_names_last(tmp_path):
    _checkpoint(tmp_path / "ckpt_final_0000100.pt")
    assert select_export_checkpoint(tmp_path).name == "ckpt_final_0000100.pt"


def test_checkpoint_manifest_schema_three_is_valid_but_wrong_kind_is_not(tmp_path):
    assert checkpoint_manifest_valid(_checkpoint(tmp_path / "a.pt", schema=3))
    assert not checkpoint_manifest_valid(_checkpoint(tmp_path / "b.pt", kind="shard"))
    assert not checkpoint_manifest_valid(tmp_path / "missing.pt")


def test_command_center_marks_training_complete_for_real_trainer_artifacts(tmp_path, monkeypatch):
    import command_center.store as store_module

    monkeypatch.setattr(store_module, "DATASETS", tmp_path)
    store = store_module.DatasetStore()
    meta = store.create("train-stage", group_config={"id": "t", "name": "t", "sources": {}})
    ckpt_dir = store.path(meta["id"]) / "output" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint(ckpt_dir / "ckpt_final.pt")

    state = store.refresh_pipeline_state(meta["id"])
    assert state["stages"]["train"] == "complete"


def test_command_center_flags_tampered_final_checkpoint_as_stale(tmp_path, monkeypatch):
    import command_center.store as store_module

    monkeypatch.setattr(store_module, "DATASETS", tmp_path)
    store = store_module.DatasetStore()
    meta = store.create("train-stale", group_config={"id": "t", "name": "t", "sources": {}})
    ckpt_dir = store.path(meta["id"]) / "output" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = _checkpoint(ckpt_dir / "ckpt_final.pt")
    final.write_bytes(b"z" * 4096)

    assert store.refresh_pipeline_state(meta["id"])["stages"]["train"] == "stale"


# --------------------------------------------------------------------------- special tokens
def _tokenizer_dir(tmp_path: Path, order: list[str]) -> Path:
    d = tmp_path / "tok"
    d.mkdir()
    (d / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {t: i for i, t in enumerate(order)}}, "added_tokens": []}), encoding="utf-8"
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<p>", "unk_token": "<u>", "bos_token": "<b>", "eos_token": "<e>"}), encoding="utf-8"
    )
    return d


def test_special_token_ids_follow_configured_order_not_hardcoded_defaults(tmp_path):
    d = _tokenizer_dir(tmp_path, ["<b>", "<e>", "<p>", "<u>", "a", "b"])
    assert special_token_ids(d) == {"bos": 0, "eos": 1, "pad": 2, "unk": 3}


def test_special_token_ids_prefer_added_tokens_and_reject_missing(tmp_path):
    d = _tokenizer_dir(tmp_path, ["<p>", "<u>", "<b>", "<e>"])
    tok = json.loads((d / "tokenizer.json").read_text())
    tok["added_tokens"] = [{"id": 9, "content": "<e>"}]
    (d / "tokenizer.json").write_text(json.dumps(tok))
    assert special_token_ids(d)["eos"] == 9

    cfg = json.loads((d / "tokenizer_config.json").read_text())
    cfg["bos_token"] = "<absent>"
    (d / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="bos_token"):
        special_token_ids(d)


def test_special_token_ids_fail_without_tokenizer_files(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        special_token_ids(tmp_path)


# --------------------------------------------------------------------------- numerics
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_loss_is_rejected_with_step(bad):
    with pytest.raises(NonFiniteLossError, match="step 7"):
        require_finite("train_loss", bad, 7)


def test_finite_loss_passes_through():
    assert require_finite("val", 1.5) == 1.5
    assert math.isfinite(require_finite("val", 0))
