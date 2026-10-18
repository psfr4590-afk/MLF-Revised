"""Canonical numeric dataset-ID identity checks for dataset sessions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(root: Path) -> dict[int, dict[str, Any]]:
    profiles_path = root / "config" / "dataset_profiles.yaml"
    groups_path = root / "config" / "dataset_groups.yaml"
    profiles_doc = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    groups_doc = yaml.safe_load(groups_path.read_text(encoding="utf-8")) or {}
    if int(profiles_doc.get("schema", -1)) != 2:
        raise ValueError("dataset_profiles.yaml must use schema 2")
    profiles = profiles_doc.get("dataset_profiles", [])
    groups = groups_doc.get("dataset_groups", [])
    group_ids = {str(item.get("id")) for item in groups if item.get("id")}
    mapping: dict[int, dict[str, Any]] = {}
    seen_groups: set[str] = set()
    for profile in profiles:
        did = int(profile["dataset_id"])
        gid = str(profile["group_id"])
        if did <= 0 or did in mapping:
            raise ValueError(f"Invalid or duplicate canonical dataset_id: {did}")
        if gid not in group_ids or gid in seen_groups:
            raise ValueError(f"Invalid or duplicate canonical group binding: {gid}")
        mapping[did] = profile
        seen_groups.add(gid)
    if sorted(mapping) != list(range(1, len(mapping) + 1)):
        raise ValueError("Canonical dataset IDs must be contiguous starting at 1")
    return mapping


def validate_session_identity(root: Path, dataset_id: int) -> dict[str, Any] | None:
    """Reject canonical dataset sessions whose numeric ID and group identity disagree."""
    catalog = load_catalog(root)
    profile = catalog.get(int(dataset_id))
    if profile is None:
        return None
    session_root = root / "datasets" / f"dataset_{int(dataset_id):03d}"
    meta_path = session_root / "dataset.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_group = str(profile["group_id"])
    if int(meta.get("id", -1)) != int(dataset_id) or meta.get("group_id") != expected_group:
        raise RuntimeError(
            f"Dataset identity mismatch for dataset_{dataset_id:03d}: "
            f"expected id={dataset_id}, group={expected_group!r}; "
            f"found id={meta.get('id')!r}, group={meta.get('group_id')!r}"
        )
    expected_profile_sha = _sha256_file(root / "config" / "dataset_profiles.yaml")
    recorded_profile_sha = (meta.get("profile") or {}).get("catalog_sha256")
    if recorded_profile_sha and recorded_profile_sha != expected_profile_sha:
        raise RuntimeError(f"Dataset profile catalog changed after dataset_{dataset_id:03d} was created")
    return profile
