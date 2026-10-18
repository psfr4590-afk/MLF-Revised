from __future__ import annotations

import json
from pathlib import Path

from pipeline.observability import classify_source_status, write_dataset_report


def test_source_status_distinguishes_configured_but_not_retrieved():
    assert classify_source_status(enabled=False, documents=0, errors=0, skipped=0, rate_limited=0) == "DISABLED"
    assert classify_source_status(enabled=True, documents=0, errors=0, skipped=0, rate_limited=0) == "EMPTY"
    assert classify_source_status(enabled=True, documents=0, errors=1, skipped=0, rate_limited=0) == "FAILED"
    assert classify_source_status(enabled=True, documents=3, errors=1, skipped=0, rate_limited=0) == "PARTIAL"
    assert classify_source_status(enabled=True, documents=0, errors=1, skipped=0, rate_limited=1) == "RATE_LIMITED"
    assert classify_source_status(enabled=True, documents=0, errors=0, skipped=2, rate_limited=0) == "SKIPPED"
    assert classify_source_status(enabled=True, documents=4, errors=0, skipped=0, rate_limited=0) == "SUCCESS"


def test_dataset_report_is_machine_and_human_readable(tmp_path: Path):
    metrics = {
        "crawl": {"document_count": 16033},
        "clean": {"document_count": 14000, "removed_count": 2033},
        "dedup": {"document_count": 13311, "removed_count": 689},
        "shard": {
            "token_count": 26374737,
            "train_document_count": 10583,
            "validation_document_count": 2728,
            "train_shards": 21,
            "validation_shards": 6,
        },
    }
    sources = [{
        "display_name": "HuggingFace",
        "status": "SUCCESS",
        "configured": True,
        "retrieved": True,
        "document_count": 5000,
    }]
    path = write_dataset_report(tmp_path / "dataset_report.json", run_id="run_test", stage_metrics=metrics, sources=sources)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset"]["crawled"] == 16033
    assert data["dataset"]["final_docs"] == 13311
    assert data["dataset"]["tokens"] == 26374737
    assert data["sources"][0]["retrieved"] is True
    assert path.with_suffix(".txt").is_file()
