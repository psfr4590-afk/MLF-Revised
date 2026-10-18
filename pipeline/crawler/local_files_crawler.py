"""Bounded local-filesystem ingestion for explicitly configured paths."""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from pathlib import Path
from typing import Iterator

from pipeline.crawler.base import BaseCrawler
from pipeline.crawler.content_parser import ContentParseError, parse_content
from pipeline.crawler.source_scorer import classify_text, classify_url, detect_code_language, score_document
from pipeline.types import Document

log = logging.getLogger("crawler.local_files")


class LocalFilesCrawler(BaseCrawler):
    """Read supported documents from configured local files/directories.

    Only explicitly configured roots are traversed. Symlinks are ignored, file
    size and document count are bounded, and unsupported/malformed files are
    skipped with observable counters rather than silently emitted as empty docs.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml",
        ".html", ".htm", ".md", ".markdown", ".txt", ".rst", ".log",
        ".docx", ".xlsx", ".xlsm", ".pptx",
    }

    def __init__(self, cfg: dict, weight_lookup, signal_tracker):
        super().__init__(cfg, weight_lookup, signal_tracker)
        self.local = cfg.get("local_files", {}) or {}
        self.max_files = int(self.local.get("max_files", 1000))
        self.max_file_bytes = int(self.local.get("max_file_bytes", 5_000_000))
        if self.max_files < 1:
            raise ValueError("crawl.local_files.max_files must be positive")
        if self.max_file_bytes < 1:
            raise ValueError("crawl.local_files.max_file_bytes must be positive")
        extensions = self.local.get("extensions") or sorted(self.SUPPORTED_EXTENSIONS)
        self.extensions = {str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in extensions}
        self.recursive = bool(self.local.get("recursive", True))
        self.paths = list(self.local.get("paths", []) or [])

    def _candidate_files(self) -> Iterator[Path]:
        seen: set[str] = set()
        emitted = 0
        project_root = Path(self.cfg.get("_project_root", Path.cwd())).resolve()
        for configured in self.paths:
            root = Path(str(configured)).expanduser()
            if not root.is_absolute():
                root = project_root / root
            if root.is_symlink():
                self.stats["skipped"] += 1
                continue
            if root.is_file():
                candidates = iter((root,))
            elif root.is_dir() and self.recursive:
                def walk_files():
                    for directory, dirnames, filenames in os.walk(root, followlinks=False):
                        dirnames[:] = sorted(
                            name for name in dirnames
                            if not (Path(directory) / name).is_symlink()
                        )
                        for filename in sorted(filenames, key=str.casefold):
                            yield Path(directory) / filename
                candidates = walk_files()
            elif root.is_dir():
                candidates = iter(sorted(root.iterdir(), key=lambda item: str(item).casefold()))
            else:
                self.stats["errors"] += 1
                log.warning("Local source path does not exist: %s", root)
                continue
            for candidate in candidates:
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in self.extensions:
                    continue
                resolved = candidate.resolve()
                identity = str(resolved).casefold()
                if identity in seen:
                    continue
                seen.add(identity)
                if emitted >= self.max_files:
                    self.stats["skipped"] += 1
                    return
                emitted += 1
                yield resolved

    def crawl(self) -> Iterator[Document]:
        for path in self._candidate_files():
            self.stats["request_count"] += 1
            try:
                size = path.stat().st_size
                if size <= 0 or size > self.max_file_bytes:
                    self.stats["skipped"] += 1
                    continue
                body = path.read_bytes()
                if len(body) > self.max_file_bytes:
                    self.stats["skipped"] += 1
                    continue
                content_type = mimetypes.guess_type(path.name)[0] or ""
                try:
                    parsed = parse_content(path.as_uri(), content_type, body)
                except ContentParseError as exc:
                    self.stats["skipped"] += 1
                    log.warning("Local document parsing failed for %s: %s", path, exc)
                    continue
                digest = hashlib.sha256(body).hexdigest()
                url = path.as_uri()
                semantic_type = classify_url(url)
                if semantic_type == "unknown":
                    semantic_type = parsed.content_type or classify_text(parsed.text)
                doc = Document(
                    doc_id=hashlib.sha256(f"{url}\0{digest}".encode("utf-8")).hexdigest(),
                    url=url,
                    source="local_files",
                    title=parsed.title or path.stem,
                    text=parsed.text,
                    content_type=semantic_type,
                    code_language=detect_code_language(path.name),
                    domain="local-filesystem",
                    meta={
                        "source_mode": "local",
                        "local_path": str(path),
                        "file_sha256": digest,
                        "file_size_bytes": len(body),
                        "parser": parsed.metadata,
                        "license": "user-provided-local-source",
                    },
                )
                doc.meta["quality_signal"] = score_document(url, doc.text)
                self.signal_tracker.record(doc.domain, doc.meta["quality_signal"])
                self.stats["successful_requests"] += 1
                yield self._apply_weights(doc)
            except OSError as exc:
                self.stats["errors"] += 1
                log.warning("Could not read local document %s: %s", path, exc)

    def observability(self) -> dict:
        result = super().observability()
        result["source_mode"] = "local"
        return result
