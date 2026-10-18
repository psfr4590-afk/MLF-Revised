"""Executable dataset identity, acquisition, rights, exclusion, and quality policy."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

SOURCE_KINDS = ("web", "github", "arxiv", "huggingface", "google")


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset contract file must contain an object: {path}")
    return raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_contract(root: Path) -> dict[str, Any]:
    return _load_yaml(root / "config" / "dataset_source_policy.yaml")


def validate_dataset_contract(root: Path, groups_path: Path, profiles_path: Path) -> dict[str, Any]:
    """Validate all ten dataset definitions before a production crawl is allowed."""
    contract = load_contract(root)
    groups_raw = _load_yaml(groups_path)
    profiles_raw = _load_yaml(profiles_path)
    groups = groups_raw.get("dataset_groups", [])
    profiles = profiles_raw.get("dataset_profiles", [])
    _require(len(groups) == 10, f"Expected 10 dataset groups, found {len(groups)}")
    _require(len(profiles) == 10, f"Expected 10 dataset profiles, found {len(profiles)}")

    group_by_id = {str(group.get("id")): group for group in groups}
    profile_by_id = {int(profile.get("dataset_id")): profile for profile in profiles}
    _require(len(group_by_id) == 10, "Dataset group IDs must be unique")
    _require(sorted(profile_by_id) == list(range(1, 11)), "Dataset profile IDs must be exactly 1..10")

    policies = contract.get("source_kinds", {})
    _require(set(policies) == set(SOURCE_KINDS), "Source policy must cover every supported source kind")

    mutable_revisions: list[str] = []
    for dataset_id in range(1, 11):
        profile = profile_by_id[dataset_id]
        group_id = str(profile.get("group_id", ""))
        _require(group_id in group_by_id, f"Dataset {dataset_id} references unknown group '{group_id}'")
        group = group_by_id[group_id]
        enabled = group.get("sources", {}) or {}
        _require(isinstance(enabled, dict), f"{group_id}.sources must be an object")
        exclusions = profile.get("exclusions", [])
        _require(isinstance(exclusions, list) and exclusions, f"Dataset {dataset_id} requires exclusions")
        for tag in exclusions:
            _require(tag in contract.get("exclusions", {}), f"Dataset {dataset_id} exclusion has no executable rule: {tag}")

        for kind in SOURCE_KINDS:
            if not bool(enabled.get(kind, False)):
                continue
            section = group.get(kind, {}) or {}
            _require(isinstance(section, dict), f"{group_id}.{kind} must be an object")
            policy = policies[kind]
            _require(policy.get("acquisition"), f"{kind} source policy is missing acquisition strategy")
            _require(policy.get("identity_fields"), f"{kind} source policy is missing identity fields")

            if kind == "web":
                seeds = section.get("seed_urls", [])
                _require(isinstance(seeds, list) and seeds, f"{group_id}.web.seed_urls must be non-empty")
                _require(all(isinstance(url, str) and url.startswith(("http://", "https://")) for url in seeds), f"{group_id}.web.seed_urls contains an invalid URL")
            elif kind == "github":
                _require(section.get("topics") or section.get("languages") or section.get("queries"), f"{group_id}.github requires topics, languages, or queries")
                _require(int(section.get("max_repos", 0)) > 0, f"{group_id}.github.max_repos must be positive")
                _require(int(section.get("min_stars", 0)) >= 0, f"{group_id}.github.min_stars must be non-negative")
            elif kind == "arxiv":
                categories = section.get("categories", [])
                _require(isinstance(categories, list) and categories, f"{group_id}.arxiv.categories must be non-empty")
                _require(0 < int(section.get("max_results", 0)) <= 200, f"{group_id}.arxiv.max_results must be 1..200")
            elif kind == "huggingface":
                datasets = section.get("datasets", [])
                _require(isinstance(datasets, list) and datasets, f"{group_id}.huggingface.datasets must be non-empty")
                for item in datasets:
                    _require(isinstance(item, dict), f"{group_id}.huggingface dataset entry must be an object")
                    for key in ("repo", "revision", "split", "max_docs"):
                        _require(item.get(key) not in (None, ""), f"{group_id}.huggingface entry missing {key}")
                    has_text_field = item.get("text_field") not in (None, "")
                    text_fields = item.get("text_fields")
                    has_text_fields = isinstance(text_fields, list) and bool(text_fields) and all(str(field).strip() for field in text_fields)
                    _require(has_text_field or has_text_fields, f"{group_id}.huggingface entry requires text_field or non-empty text_fields")
                    _require(int(item["max_docs"]) > 0, f"{group_id}.huggingface.max_docs must be positive")
                    if str(item["revision"]) in {"main", "master", "HEAD"}:
                        mutable_revisions.append(f"{group_id}:huggingface:{item['repo']}")
            elif kind == "google":
                queries = section.get("queries", [])
                _require(isinstance(queries, list) and queries, f"{group_id}.google.queries must be non-empty")

    return {
        "schema": int(contract.get("schema", 0)),
        "groups": 10,
        "profiles": 10,
        "source_kinds": list(SOURCE_KINDS),
        "executable_exclusion_tags": sorted(contract.get("exclusions", {})),
        "mutable_revisions": mutable_revisions,
        "production_revision_ready": not mutable_revisions,
    }


def compile_exclusion_patterns(root: Path, tags: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    contract = load_contract(root)
    rules = contract.get("exclusions", {})
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for tag in tags:
        for pattern in rules.get(tag, {}).get("patterns", []):
            compiled.append((tag, re.compile(pattern, re.IGNORECASE | re.DOTALL)))
    return compiled


def evaluate_document(root: Path, doc: Any, exclusions: list[str]) -> tuple[bool, list[str], float]:
    """Return keep/reject, matched exclusion tags, and the document quality score."""
    text = str(getattr(doc, "text", "") or "")
    meta = getattr(doc, "meta", {}) or {}
    quality = float(meta.get("quality_signal", getattr(doc, "quality_score", 0.0)) or 0.0)
    contract = load_contract(root)
    quality_cfg = contract.get("quality", {})
    min_words = int(quality_cfg.get("min_text_words", 30))
    min_quality = float(quality_cfg.get("min_quality_score", 0.35))
    matches = [tag for tag, pattern in compile_exclusion_patterns(root, exclusions) if pattern.search(text)]
    keep = bool(text.strip()) and len(text.split()) >= min_words and quality >= min_quality and not matches
    return keep, sorted(set(matches)), quality


def distribution_ready(manifest: dict[str, Any]) -> bool:
    """Unknown rights are explicitly allowed during collection but block distribution."""
    rights = manifest.get("rights", {})
    if not isinstance(rights, dict):
        return False
    return bool(rights.get("distribution_ready", False))
