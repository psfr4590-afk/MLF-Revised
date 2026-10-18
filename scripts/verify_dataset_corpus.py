"""Validate a crawled dataset corpus before clean/dedup/training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pipeline.dataset_contract import evaluate_document, validate_dataset_contract
from pipeline.types import Document

ROOT = Path(__file__).resolve().parents[1]


def _identity_is_complete(doc: Document) -> bool:
    identity = doc.meta.get("source_identity")
    if doc.source == "web":
        return bool(doc.url)
    if not isinstance(identity, dict):
        return False
    required = {
        "github": ("full_name", "commit_sha"),
        "arxiv": ("arxiv_id", "version"),
        "huggingface": ("repo", "revision", "split", "row_index"),
        "google": ("result_url", "query"),
    }.get(doc.source, ())
    return all(identity.get(key) not in (None, "") for key in required)


def main() -> int:
    parser = argparse.ArgumentParser(description="Accept or reject a crawled Model Lab dataset corpus")
    parser.add_argument("--input", required=True, help="Crawled JSONL file")
    parser.add_argument("--dataset-id", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--require-rights", action="store_true", help="Fail if any source lacks verified rights metadata")
    parser.add_argument("--require-immutable-revisions", action="store_true", help="Fail if a configured Hugging Face source is mutable")
    args = parser.parse_args()

    report = validate_dataset_contract(ROOT, ROOT / "config" / "dataset_groups.yaml", ROOT / "config" / "dataset_profiles.yaml")
    if args.require_immutable_revisions and not report["production_revision_ready"]:
        print("DATASET CORPUS BLOCKED: one or more configured sources use mutable revisions")
        for item in report["mutable_revisions"]:
            print(f"  - {item}")
        return 1

    profiles = yaml.safe_load((ROOT / "config" / "dataset_profiles.yaml").read_text(encoding="utf-8")) or {}
    profile = next(item for item in profiles["dataset_profiles"] if int(item["dataset_id"]) == args.dataset_id)
    expected_group = str(profile["group_id"])

    seen: set[str] = set()
    rows = 0
    rejected = 0
    rights_unknown = 0
    identity_missing = 0
    reasons: dict[str, int] = {}

    with Path(args.input).open("r", encoding="utf-8") as handle:
        for _line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            doc = Document.from_dict(record)
            if doc.doc_id in seen:
                rejected += 1
                reasons["duplicate_document_id"] = reasons.get("duplicate_document_id", 0) + 1
                continue
            seen.add(doc.doc_id)
            if str(doc.meta.get("dataset_group", "")) != expected_group:
                rejected += 1
                reasons["dataset_group_mismatch"] = reasons.get("dataset_group_mismatch", 0) + 1
                continue
            if doc.source not in {"web", "github", "arxiv", "huggingface", "google"}:
                rejected += 1
                reasons["unsupported_source"] = reasons.get("unsupported_source", 0) + 1
                continue
            keep, matches, _ = evaluate_document(ROOT, doc, profile.get("exclusions", []))
            if not keep:
                rejected += 1
                key = ",".join(matches) if matches else "quality_gate"
                reasons[key] = reasons.get(key, 0) + 1
                continue
            if not _identity_is_complete(doc):
                rejected += 1
                identity_missing += 1
                reasons["missing_source_identity"] = reasons.get("missing_source_identity", 0) + 1
                continue
            if str(doc.meta.get("license_status", "unknown")) not in {"verified", "permissive", "approved"}:
                rights_unknown += 1

    accepted = rows - rejected
    status = accepted > 0 and rejected == 0 and (not args.require_rights or rights_unknown == 0)
    print(json.dumps({
        "status": "passed" if status else "failed",
        "dataset_id": args.dataset_id,
        "dataset_group": expected_group,
        "rows": rows,
        "accepted": accepted,
        "rejected": rejected,
        "rights_unknown": rights_unknown,
        "identity_missing": identity_missing,
        "reasons": reasons,
        "distribution_ready": rights_unknown == 0 and report["production_revision_ready"],
        "production_revision_ready": report["production_revision_ready"],
    }, indent=2, sort_keys=True))
    if not status:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
