from pathlib import Path

import yaml

from pipeline.config_validation import validate_config

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_default_config_is_bounded_and_cpu_safe():
    cfg = _load("pipeline_config.yaml")
    validate_config(cfg)
    assert cfg["train"]["total_steps"] == 2
    assert cfg["train"]["allow_cpu_training"] is True
    assert cfg["stages"]["export"] is False
    assert cfg["crawl"]["max_pages_per_domain"] <= 2


def test_full_profile_is_explicitly_large():
    cfg = _load("pipeline_config.full.yaml")
    validate_config(cfg)
    assert cfg["train"]["total_steps"] >= 100000
    assert cfg["train"]["allow_cpu_training"] is False
    assert cfg["stages"]["export"] is True


def test_default_and_full_profiles_are_distinct():
    starter = _load("pipeline_config.yaml")
    full = _load("pipeline_config.full.yaml")
    assert starter["pipeline"]["name"] != full["pipeline"]["name"]
    assert starter["pipeline"]["model_name"] != full["pipeline"]["model_name"]
