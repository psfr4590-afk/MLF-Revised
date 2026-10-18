from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.dataset_contract import evaluate_document, validate_dataset_contract
from pipeline.types import Document

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT / "config" / "dataset_groups.yaml"
PROFILES = ROOT / "config" / "dataset_profiles.yaml"


def test_all_ten_dataset_contracts_validate():
    report = validate_dataset_contract(ROOT, GROUPS, PROFILES)
    assert report["groups"] == 10
    assert report["profiles"] == 10
    assert set(report["source_kinds"]) == {"web", "github", "arxiv", "huggingface", "google"}
    assert report["executable_exclusion_tags"]
    assert report["mutable_revisions"] == []
    assert report["production_revision_ready"] is True


def test_huggingface_revision_is_required(tmp_path):
    groups = tmp_path / "groups.yaml"
    profiles = tmp_path / "profiles.yaml"
    data = yaml.safe_load(GROUPS.read_text(encoding="utf-8"))
    for group in data["dataset_groups"]:
        for dataset in group.get("huggingface", {}).get("datasets", []):
            dataset["revision"] = ""
            break
        else:
            continue
        break
    groups.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    profiles.write_text(PROFILES.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="missing revision"):
        validate_dataset_contract(ROOT, groups, profiles)


def test_profile_exclusion_is_executable():
    doc = Document(
        doc_id="test:excluded",
        url="https://example.test",
        source="web",
        text="This is a sufficiently long technical document. Vote for this candidate and support this candidate because everyone must vote for them. " * 2,
        title="test",
        content_type="technical_doc",
        domain="example.test",
        meta={"quality_signal": 0.9},
    )
    keep, matches, quality = evaluate_document(ROOT, doc, ["political-persuasion"])
    assert not keep
    assert "political-persuasion" in matches
    assert quality == 0.9


def test_quality_gate_rejects_short_documents():
    doc = Document(
        doc_id="test:short",
        url="https://example.test",
        source="web",
        text="too short",
        title="test",
        content_type="technical_doc",
        domain="example.test",
        meta={"quality_signal": 0.9},
    )
    keep, matches, _ = evaluate_document(ROOT, doc, [])
    assert not keep
    assert matches == []
