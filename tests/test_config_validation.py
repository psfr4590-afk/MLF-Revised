from pathlib import Path

import pytest
import yaml

from pipeline.config_validation import validate_config

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_main_config_is_validator_clean():
    validate_config(load_yaml(ROOT / "config" / "pipeline_config.yaml"))


def test_smoke_config_is_validator_clean():
    validate_config(load_yaml(ROOT / "config" / "pipeline_config.smoke.yaml"))


def test_export_all_quant_is_accepted():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.smoke.yaml")
    cfg["export"]["quant"] = "ALL"
    validate_config(cfg)


def test_pipeline_level_model_preset_is_rejected():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.yaml")
    cfg["pipeline"]["model_preset"] = "85M"
    with pytest.raises(ValueError, match="Unsupported pipeline configuration keys: model_preset"):
        validate_config(cfg)


def test_unknown_pipeline_stage_is_rejected():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.yaml")
    cfg["stages"]["shardd"] = True
    with pytest.raises(ValueError, match="Unsupported stages configuration keys: shardd"):
        validate_config(cfg)


def test_non_boolean_pipeline_stage_is_rejected():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.yaml")
    cfg["stages"]["train"] = "true"
    with pytest.raises(ValueError, match="Pipeline stage flags must be boolean"):
        validate_config(cfg)


def test_unknown_section_key_is_rejected():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.yaml")
    cfg["train"]["lr_mxa"] = 0.0003
    with pytest.raises(ValueError, match="Unsupported train configuration keys: lr_mxa"):
        validate_config(cfg)


def test_cross_section_vocab_invariant_is_rejected():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.yaml")
    cfg["train"]["vocab_size"] = 1024
    with pytest.raises(ValueError, match="tokenizer.vocab_size must equal train.vocab_size"):
        validate_config(cfg)


def test_cross_section_sequence_invariant_is_rejected():
    cfg = load_yaml(ROOT / "config" / "pipeline_config.yaml")
    cfg["train"]["seq_len"] = 64
    with pytest.raises(ValueError, match="shard.sequence_length must equal train.seq_len"):
        validate_config(cfg)
