"""Dataset and source observability helpers for Model Lab."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_STATUSES = {
    "SUCCESS",
    "PARTIAL",
    "FAILED",
    "DISABLED",
    "RATE_LIMITED",
    "EMPTY",
    "SKIPPED",
}

STAGE_NAMES = ("crawl", "clean", "dedup", "weight", "tokenize", "shard")


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _jsonl_stats(path: Path) -> dict[str, int]:
    documents = 0
    words = 0
    characters = 0
    if not path.is_file():
        return {"documents": 0, "words": 0, "characters": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", "") or "")
            documents += 1
            words += len(text.split())
            characters += len(text)
    return {"documents": documents, "words": words, "characters": characters}


def stage_counts(input_path: Path | None, output_path: Path | None) -> dict[str, int | None]:
    before = _jsonl_stats(input_path) if input_path else {"documents": 0, "words": 0, "characters": 0}
    after = _jsonl_stats(output_path) if output_path else {"documents": 0, "words": 0, "characters": 0}
    return {
        "input_document_count": before["documents"] if input_path else None,
        "document_count": after["documents"] if output_path else None,
        "removed_count": max(0, before["documents"] - after["documents"]) if input_path and output_path else None,
        "input_word_count": before["words"] if input_path else None,
        "word_count": after["words"] if output_path else None,
    }


def classify_source_status(*, enabled: bool, documents: int, errors: int, skipped: int, rate_limited: int) -> str:
    if not enabled:
        return "DISABLED"
    if rate_limited:
        return "RATE_LIMITED"
    if documents and errors:
        return "PARTIAL"
    if documents:
        return "SUCCESS"
    if errors:
        return "FAILED"
    if skipped:
        return "SKIPPED"
    return "EMPTY"


def throughput(count: int | None, seconds: float) -> float | None:
    if count is None or seconds <= 0:
        return None
    return round(count / seconds, 3)


def source_record(
    *,
    configured: dict[str, Any],
    retrieved: bool,
    status: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    record = dict(configured)
    record.update({
        "status": status,
        "configured": True,
        "retrieved": retrieved,
        "request_count": int(stats.get("request_count", 0)),
        "successful_requests": int(stats.get("successful_requests", 0)),
        "failed_requests": int(stats.get("failed_requests", 0)),
        "http_status_codes": dict(stats.get("http_status_codes", {})),
        "document_count": int(stats.get("document_count", 0)),
        "duration_seconds": round(float(stats.get("duration_seconds", 0.0)), 6),
        "retry_count": int(stats.get("retry_count", 0)),
        "revision": stats.get("revision", configured.get("revision")),
        "source_hash": stats.get("source_hash"),
        "license_status": stats.get("license_status", configured.get("license", "unknown")),
    })
    return record


def write_dataset_report(
    path: Path,
    *,
    run_id: str,
    stage_metrics: dict[str, dict[str, Any]],
    sources: Iterable[dict[str, Any]],
) -> Path:
    stages = {name: stage_metrics.get(name, {}) for name in STAGE_NAMES}
    shard = stages.get("shard", {})
    report = {
        "schema": 1,
        "run_id": run_id,
        "generated_at": utc_now(),
        "dataset": {
            "crawled": stages["crawl"].get("document_count"),
            "final_docs": stages["dedup"].get("document_count") or stages["clean"].get("document_count"),
            "tokens": shard.get("token_count"),
            "train_docs": shard.get("train_document_count"),
            "validation_docs": shard.get("validation_document_count"),
            "train_shards": shard.get("train_shards"),
            "validation_shards": shard.get("validation_shards"),
        },
        "stages": stages,
        "sources": list(sources),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    d = report["dataset"]
    lines = [
        "DATASET",
        "-------",
        f"Crawled:        {d['crawled'] if d['crawled'] is not None else 'N/A'}",
        f"Final docs:     {d['final_docs'] if d['final_docs'] is not None else 'N/A'}",
        f"Tokens:         {d['tokens'] if d['tokens'] is not None else 'N/A'}",
        f"Train docs:     {d['train_docs'] if d['train_docs'] is not None else 'N/A'}",
        f"Validation:     {d['validation_docs'] if d['validation_docs'] is not None else 'N/A'}",
        f"Train shards:   {d['train_shards'] if d['train_shards'] is not None else 'N/A'}",
        f"Validation:     {d['validation_shards'] if d['validation_shards'] is not None else 'N/A'}",
        "",
        "Sources:",
    ]
    for source in report["sources"]:
        label = source.get("display_name") or source.get("identifier") or source.get("kind", "unknown")
        lines.append(f"{label:<16} {source.get('status', 'UNKNOWN')}")
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
