"""Abstract base class for all crawlers."""
from __future__ import annotations

import abc
import hashlib
import json
import logging
from typing import Iterator

from pipeline.types import Document

log = logging.getLogger("crawler.base")


class BaseCrawler(abc.ABC):
    """Every crawler yields Document objects with deterministic retrieval identity."""

    def __init__(self, cfg: dict, weight_lookup, signal_tracker):
        self.cfg = cfg
        self.weight_lookup = weight_lookup
        self.signal_tracker = signal_tracker
        self.stats = {
            "fetched": 0,
            "skipped": 0,
            "errors": 0,
            "abandoned_domains": 0,
            "request_count": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "http_status_codes": {},
            "retry_count": 0,
            "rate_limited": 0,
            "document_count": 0,
            "revisions": set(),
            "license_statuses": set(),
        }
        self._source_hash = hashlib.sha256()

    @abc.abstractmethod
    def crawl(self) -> Iterator[Document]:
        ...

    def record_response(self, response) -> None:
        self.stats["request_count"] += 1
        status = int(getattr(response, "status_code", 0) or 0)
        codes = self.stats["http_status_codes"]
        key = str(status)
        codes[key] = int(codes.get(key, 0)) + 1
        if 200 <= status < 400:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        if status == 429:
            self.stats["rate_limited"] += 1

    def record_request_error(self) -> None:
        self.stats["request_count"] += 1
        self.stats["failed_requests"] += 1

    def record_retry(self) -> None:
        self.stats["retry_count"] += 1

    def record_document(self, doc: Document) -> Document:
        self.stats["document_count"] += 1
        revision = doc.meta.get("revision")
        source_identity = doc.meta.get("source_identity", {})
        revision = revision or source_identity.get("revision") or source_identity.get("commit_sha") or source_identity.get("version")
        if revision:
            self.stats["revisions"].add(str(revision))
        license_status = doc.meta.get("license_status") or doc.meta.get("license") or "unknown"
        self.stats["license_statuses"].add(str(license_status))
        record = self._record_retrieval_identity(doc)
        self._source_hash.update(json.dumps(record.to_jsonl(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        self._source_hash.update(b"\n")
        return record

    def observability(self) -> dict:
        return {
            "request_count": self.stats["request_count"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "http_status_codes": dict(self.stats["http_status_codes"]),
            "retry_count": self.stats["retry_count"],
            "rate_limited": self.stats["rate_limited"],
            "document_count": self.stats["document_count"],
            "revision": sorted(self.stats["revisions"])[0] if len(self.stats["revisions"]) == 1 else (sorted(self.stats["revisions"]) if self.stats["revisions"] else None),
            "license_status": sorted(self.stats["license_statuses"])[0] if len(self.stats["license_statuses"]) == 1 else (sorted(self.stats["license_statuses"]) if self.stats["license_statuses"] else "unknown"),
            "source_hash": self._source_hash.hexdigest(),
        }

    @staticmethod
    def _record_retrieval_identity(doc: Document) -> Document:
        """Bind the emitted source record to the exact canonical bytes consumed downstream."""
        canonical = json.dumps(
            doc.to_jsonl(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        doc.meta["retrieval_provenance"] = {
            "identity_type": "canonical_record",
            "record_sha256": hashlib.sha256(canonical).hexdigest(),
        }
        return doc

    def _apply_weights(self, doc: Document) -> Document:
        dw, category = self.weight_lookup.domain_weight(doc.domain)
        ctw = self.weight_lookup.content_type_weight(doc.content_type)
        qs = self.weight_lookup.quality_score(doc.text)
        doc.domain_weight = dw
        doc.content_type_weight = ctw
        doc.quality_score = qs
        doc.meta["quality_signal"] = qs
        if doc.content_type == "source_code" and doc.code_language:
            ctw *= self.weight_lookup.code_language_weight(doc.code_language)
        doc.content_type_weight = ctw
        doc.final_weight = dw * ctw * qs
        if not doc.meta.get("category"):
            doc.meta["category"] = category
        return self.record_document(doc)

    def print_stats(self):
        s = self.stats
        log.info(
            f"{self.__class__.__name__} | fetched={s['fetched']} "
            f"skipped={s['skipped']} errors={s['errors']} "
            f"abandoned_domains={s['abandoned_domains']}"
        )
