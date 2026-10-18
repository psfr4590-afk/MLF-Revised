from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


def load_pipeline_config():
    p = ROOT / "config" / "pipeline_config.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def load_groups():
    p = ROOT / "config" / "dataset_groups.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    groups = raw.get("dataset_groups", [])
    if not isinstance(groups, list):
        raise ValueError("config/dataset_groups.yaml must contain a dataset_groups list")
    return groups


def load_profiles():
    p = ROOT / "config" / "dataset_profiles.yaml"
    if not p.is_file():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if int(raw.get("schema", -1)) != 2:
        raise ValueError("config/dataset_profiles.yaml must use schema 2")
    profiles = raw.get("dataset_profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("config/dataset_profiles.yaml must contain a dataset_profiles list")
    return profiles


def profile_by_id(did):
    return next((p for p in load_profiles() if int(p.get("dataset_id", -1)) == int(did)), None)


def profile_by_group_id(gid):
    return next((p for p in load_profiles() if p.get("group_id") == gid), None)


def validate_profile_catalog() -> dict[int, dict]:
    """Validate the stable numeric dataset-ID to canonical-group contract."""
    groups = load_groups()
    group_ids = {str(g.get("id")) for g in groups if g.get("id")}
    profiles = load_profiles()
    seen_ids: set[int] = set()
    seen_groups: set[str] = set()
    mapping: dict[int, dict] = {}
    for profile in profiles:
        try:
            did = int(profile["dataset_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every dataset profile requires an integer dataset_id") from exc
        gid = str(profile.get("group_id", ""))
        if did <= 0:
            raise ValueError(f"Dataset profile ID must be positive: {did}")
        if did in seen_ids:
            raise ValueError(f"Duplicate canonical dataset_id: {did}")
        if not gid or gid not in group_ids:
            raise ValueError(f"Dataset profile {did} references unknown canonical group: {gid!r}")
        if gid in seen_groups:
            raise ValueError(f"Canonical group is bound to more than one dataset_id: {gid}")
        seen_ids.add(did)
        seen_groups.add(gid)
        mapping[did] = profile
    if profiles and sorted(seen_ids) != list(range(1, len(profiles) + 1)):
        raise ValueError("Canonical dataset IDs must be contiguous starting at 1")
    return mapping


def group_by_id(gid):
    return next((g for g in load_groups() if g.get("id") == gid), None)
