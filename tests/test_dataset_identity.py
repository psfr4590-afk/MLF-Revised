from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from command_center.store import DatasetStore
from pipeline.config_validation import validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_all_checked_in_pipeline_configs_validate():
    for name in ("pipeline_config.yaml", "pipeline_config.smoke.yaml", "pipeline_config.full.yaml", "pipeline_config.dataset.yaml"):
        cfg = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
        validate_config(cfg)


def test_dataset_session_config_declares_profile_catalog():
    cfg = yaml.safe_load((ROOT / "config" / "pipeline_config.dataset.yaml").read_text(encoding="utf-8"))
    assert cfg["crawl"]["dataset_profiles_file"] == "config/dataset_profiles.yaml"
    assert (ROOT / cfg["crawl"]["dataset_profiles_file"]).is_file()


def test_canonical_seed_ids_are_stable(tmp_path, monkeypatch):
    import command_center.store as store_module

    monkeypatch.setattr(store_module, "DATASETS", tmp_path / "datasets")
    store = DatasetStore()
    datasets = store.ensure_seed_datasets()
    canonical = {item["group_id"]: item["id"] for item in datasets if item.get("profile")}
    assert canonical == {
        "swe_cs_systems": 1,
        "ai_ml_cybersec_dataeng": 2,
        "sci_reasoning_forensics_formal": 3,
        "domain_finance_bio_robotics": 4,
        "math_statistics_optimization": 5,
        "physics_chemistry_materials": 6,
        "biomedical_health_science": 7,
        "law_compliance_governance": 8,
        "linguistics_information_retrieval": 9,
        "climate_energy_geospatial": 10,
    }


def test_canonical_group_cannot_be_reassigned_to_another_numeric_id(tmp_path, monkeypatch):
    import command_center.store as store_module

    monkeypatch.setattr(store_module, "DATASETS", tmp_path / "datasets")
    store = DatasetStore()
    root = store.path(1)
    root.mkdir(parents=True)
    (root / "dataset.json").write_text(json.dumps({"id": 1, "group_id": "wrong_group"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        store.ensure_seed_datasets()


def test_canonical_dataset_cannot_be_duplicated(tmp_path, monkeypatch):
    import command_center.store as store_module

    monkeypatch.setattr(store_module, "DATASETS", tmp_path / "datasets")
    store = DatasetStore()
    store.create("Systems", group_id="swe_cs_systems")
    with pytest.raises(ValueError, match="already exists"):
        store.create("Systems duplicate", group_id="swe_cs_systems")
