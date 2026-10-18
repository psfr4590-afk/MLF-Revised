from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_release
from scripts.export_cards import write_export_cards


def _manifest(path: Path, rows: int, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps({
            "schema": 2,
            "kind": stage,
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
            "provenance": {"pipeline_config_sha256": "config-sha"},
        }),
        encoding="utf-8",
    )


def test_write_export_cards_creates_both_cards_and_lineage(tmp_path, monkeypatch):
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    for prefix, rows in (("01_crawled", 10), ("02_cleaned", 8), ("03_deduped", 7), ("04_weighted", 9)):
        _manifest(scratch / f"{prefix}.jsonl", rows, prefix.split("_", 1)[1])
    (output / "tokenizer").mkdir(parents=True)
    (output / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (output / "shards").mkdir(parents=True)
    (output / "shards" / "shards.manifest.json").write_text(
        json.dumps({"dtype": "uint16", "files": [{"name": "shard_00000_train.bin", "size": 20}]}),
        encoding="utf-8",
    )
    (tmp_path / "source_manifest.json").write_text(
        json.dumps({"schema": 1, "sources": [], "source_definition_files": {}, "retrieval_started_at": "test", "rights_note": "test"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.export_cards._git_commit", lambda: "git-sha")

    manifest = {
        "model_name": "test-model",
        "quantization": "F16",
        "checkpoint_step": 3,
        "final_gguf": str(output / "gguf" / "model-f16.gguf"),
        "final_gguf_sha256": "gguf-sha",
        "f16_gguf_sha256": "gguf-sha",
        "converter_version": "b10516@commit",
        "model_config": {"vocab_size": 32},
    }
    paths = write_export_cards(
        output,
        manifest,
        {"step": 3, "model_cfg": {"vocab_size": 32}, "train_cfg": {"seed": 42}, "provenance": {"shard_manifest_sha256": "shard-sha"}},
    )

    dataset_card = Path(paths["dataset_card"])
    model_card = Path(paths["model_card"])
    assert dataset_card.is_file()
    assert model_card.is_file()
    assert "Pipeline configuration SHA-256 (legacy compatibility evidence): `config-sha`" in dataset_card.read_text(encoding="utf-8")
    assert "Final GGUF SHA-256: `gguf-sha`" in model_card.read_text(encoding="utf-8")
    assert "Dataset card: `DATASET_CARD.md`" in model_card.read_text(encoding="utf-8")


def test_write_export_cards_refuses_missing_source_manifest(tmp_path):
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Source manifest is mandatory"):
        write_export_cards(output, {}, {})

    assert not (output / "gguf" / "DATASET_CARD.md").exists()
    assert not (output / "gguf" / "MODEL_CARD.md").exists()


def test_release_verifier_rejects_missing_empty_and_modified_cards(tmp_path):
    dataset_card = tmp_path / "DATASET_CARD.md"
    model_card = tmp_path / "MODEL_CARD.md"
    dataset_card.write_text("# Dataset Card\n", encoding="utf-8")
    model_card.write_text("# Model Card\n", encoding="utf-8")
    manifest = {
        "dataset_card": str(dataset_card),
        "model_card": str(model_card),
        "dataset_card_sha256": hashlib.sha256(dataset_card.read_bytes()).hexdigest(),
        "model_card_sha256": hashlib.sha256(model_card.read_bytes()).hexdigest(),
    }

    verify_release._validate_export_cards(manifest)

    missing = dict(manifest)
    missing["dataset_card"] = str(tmp_path / "missing.md")
    with pytest.raises(RuntimeError, match="Missing or empty export card"):
        verify_release._validate_export_cards(missing)

    dataset_card.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing or empty export card"):
        verify_release._validate_export_cards(manifest)

    dataset_card.write_text("# Dataset Card modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash missing or mismatched"):
        verify_release._validate_export_cards(manifest)


def test_release_verifier_rejects_card_without_recorded_hash(tmp_path):
    dataset_card = tmp_path / "DATASET_CARD.md"
    model_card = tmp_path / "MODEL_CARD.md"
    dataset_card.write_text("# Dataset Card\n", encoding="utf-8")
    model_card.write_text("# Model Card\n", encoding="utf-8")
    manifest = {
        "dataset_card": str(dataset_card),
        "model_card": str(model_card),
        "dataset_card_sha256": hashlib.sha256(dataset_card.read_bytes()).hexdigest(),
    }

    with pytest.raises(RuntimeError, match="hash missing or mismatched"):
        verify_release._validate_export_cards(manifest)
