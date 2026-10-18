from __future__ import annotations

import hashlib
import json

from pipeline.crawler.base import BaseCrawler
from pipeline.types import Document


def test_crawler_record_identity_is_deterministic():
    doc = Document(
        doc_id="source-1",
        url="https://example.com/item",
        source="web",
        text="deterministic source record",
        title="Example",
        content_type="documentation",
        domain="example.com",
    )

    BaseCrawler._record_retrieval_identity(doc)

    canonical = json.dumps(
        doc.to_jsonl() | {"meta": {k: v for k, v in doc.meta.items() if k != "retrieval_provenance"}},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()

    assert doc.meta["retrieval_provenance"] == {
        "identity_type": "canonical_record",
        "record_sha256": expected,
    }


def test_crawler_record_identity_changes_when_source_record_changes():
    first = Document(doc_id="source-1", url="https://example.com/item", text="one")
    second = Document(doc_id="source-1", url="https://example.com/item", text="two")

    BaseCrawler._record_retrieval_identity(first)
    BaseCrawler._record_retrieval_identity(second)

    assert first.meta["retrieval_provenance"]["record_sha256"] != second.meta["retrieval_provenance"]["record_sha256"]
