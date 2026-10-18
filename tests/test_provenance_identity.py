from __future__ import annotations

import json
from pathlib import Path

import yaml

from pipeline.orchestrator import (
    _implementation_sha256,
    _source_definition_paths,
    _source_manifest_is_valid,
    _write_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_GROUPS = {
    "swe_cs_systems",
    "ai_ml_cybersec_dataeng",
    "sci_reasoning_forensics_formal",
    "domain_finance_bio_robotics",
    "math_statistics_optimization",
    "physics_chemistry_materials",
    "biomedical_health_science",
    "law_compliance_governance",
    "linguistics_information_retrieval",
    "climate_energy_geospatial",
}


def _load_config(name: str) -> dict:
    path = ROOT / "config" / name
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_pipeline_config_path"] = str(path)
    return cfg


def test_source_definition_hashes_include_seed_urls():
    cfg = _load_config("pipeline_config.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    assert "seed_urls" in paths
    assert paths["seed_urls"].is_file()


def test_source_definition_paths_include_dataset_profile_catalog():
    cfg = _load_config("pipeline_config.dataset.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    assert paths["dataset_profiles"] == (ROOT / "config" / "dataset_profiles.yaml").resolve()
    assert paths["dataset_profiles"].is_file()


def test_source_definition_paths_exclude_training_pipeline_config():
    cfg = _load_config("pipeline_config.smoke.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    assert "pipeline_config" not in paths


def test_dataset_session_config_binds_to_canonical_production_groups():
    cfg = _load_config("pipeline_config.dataset.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    assert paths["dataset_groups"] == (ROOT / "config" / "dataset_groups.yaml").resolve()
    assert paths["dataset_groups"].is_file()
    assert cfg["crawl"]["sources"] == {
        "web": True,
        "github": True,
        "arxiv": True,
        "huggingface": True,
        "google": True,
    }
    groups = yaml.safe_load(paths["dataset_groups"].read_text(encoding="utf-8"))["dataset_groups"]
    group_ids = {str(group["id"]) for group in groups}
    assert group_ids == EXPECTED_GROUPS
    assert len(groups) == 10


def test_dataset_profiles_schema2_bind_numeric_ids_to_canonical_groups():
    path = ROOT / "config" / "dataset_profiles.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["schema"] == 2
    profiles = data["dataset_profiles"]
    assert len(profiles) == 10
    assert [item["dataset_id"] for item in profiles] == list(range(1, 11))
    assert {item["group_id"] for item in profiles} == EXPECTED_GROUPS
    assert len({item["group_id"] for item in profiles}) == len(profiles)


def test_source_manifest_records_complete_release_metadata(tmp_path):
    cfg = _load_config("pipeline_config.full.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    target = tmp_path / "source_manifest.json"
    _write_source_manifest(
        target,
        cfg,
        paths,
        selected_group_id="swe_cs_systems",
        retrieval_started_at="2026-01-01T00:00:00+00:00",
        retrieval_completed_at="2026-01-01T00:01:00+00:00",
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema"] == 3
    assert data["dataset_group"] == "swe_cs_systems"
    assert data["retrieval_started_at"]
    assert data["retrieval_completed_at"]
    assert "pipeline_config" not in data["source_definition_files"]
    assert data["source_definition_files"]["dataset_profiles"]["sha256"]
    assert data["sources"]
    group = next(item for item in yaml.safe_load((ROOT / "config/dataset_groups.yaml").read_text(encoding="utf-8"))["dataset_groups"] if item["id"] == "swe_cs_systems")
    expected_kinds = {kind for kind, enabled in group.get("sources", {}).items() if enabled}
    assert {source["kind"] for source in data["sources"]} == expected_kinds
    assert {"github", "google"} - expected_kinds == {"github", "google"}
    assert expected_kinds == {"web", "arxiv", "huggingface"}
    assert {source["dataset_group"] for source in data["sources"]} == {"swe_cs_systems"}
    assert all({"kind", "identifier", "revision", "license", "raw_source_sha256", "dataset_group"} <= set(source) for source in data["sources"])
    assert _source_manifest_is_valid(target)
    assert _source_manifest_is_valid(target, "swe_cs_systems", paths)
    assert not _source_manifest_is_valid(target, "ai_ml_cybersec_dataeng", paths)


def test_source_manifest_all_groups_preserves_group_identity(tmp_path):
    cfg = _load_config("pipeline_config.full.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    target = tmp_path / "source_manifest.json"
    _write_source_manifest(
        target,
        cfg,
        paths,
        selected_group_id="all",
        retrieval_started_at="2026-01-01T00:00:00+00:00",
        retrieval_completed_at="2026-01-01T00:01:00+00:00",
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["dataset_group"] == "all"
    groups = {source["dataset_group"] for source in data["sources"]}
    assert groups == EXPECTED_GROUPS
    assert _source_manifest_is_valid(target, "all", paths)


def test_source_manifest_validation_rejects_incomplete_manifest(tmp_path):
    target = tmp_path / "source_manifest.json"
    target.write_text(json.dumps({"schema": 1, "sources": []}), encoding="utf-8")
    assert not _source_manifest_is_valid(target)


def test_source_manifest_validation_rejects_changed_definition_hash(tmp_path):
    cfg = _load_config("pipeline_config.full.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    target = tmp_path / "source_manifest.json"
    _write_source_manifest(
        target,
        cfg,
        paths,
        selected_group_id="swe_cs_systems",
        retrieval_started_at="2026-01-01T00:00:00+00:00",
        retrieval_completed_at="2026-01-01T00:01:00+00:00",
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    data["source_definition_files"]["dataset_groups"]["sha256"] = "0" * 64
    target.write_text(json.dumps(data), encoding="utf-8")
    assert not _source_manifest_is_valid(target, "swe_cs_systems", paths)


def test_source_manifest_validation_rejects_changed_dataset_profile_catalog(tmp_path):
    cfg = _load_config("pipeline_config.dataset.yaml")
    paths = _source_definition_paths(cfg, ROOT)
    target = tmp_path / "source_manifest.json"
    _write_source_manifest(
        target,
        cfg,
        paths,
        selected_group_id="swe_cs_systems",
        retrieval_started_at="2026-01-01T00:00:00+00:00",
        retrieval_completed_at="2026-01-01T00:01:00+00:00",
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    data["source_definition_files"]["dataset_profiles"]["sha256"] = "0" * 64
    target.write_text(json.dumps(data), encoding="utf-8")
    assert not _source_manifest_is_valid(target, "swe_cs_systems", paths)


def test_stage_implementation_identity_is_nonempty():
    digest = _implementation_sha256("clean")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
