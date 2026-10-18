"""ArXiv Atom-feed crawler with bounded result counts."""
from __future__ import annotations

import logging
import re
from defusedxml import ElementTree as ET
from typing import Iterator

import requests

from pipeline.crawler.base import BaseCrawler
from pipeline.types import Document

log = logging.getLogger("crawler.arxiv")
ATOM = "http://www.w3.org/2005/Atom"


class ArxivCrawler(BaseCrawler):
    def __init__(self, cfg: dict, weight_lookup, signal_tracker):
        super().__init__(cfg, weight_lookup, signal_tracker)
        self.cfg_arxiv = cfg.get("arxiv", {})
        self.timeout = int(cfg.get("timeout", 20))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ML-Framework/1.3"

    def crawl(self) -> Iterator[Document]:
        categories = self.cfg_arxiv.get("categories", ["cs.AI", "cs.LG", "cs.CL"])
        max_results = min(200, int(self.cfg_arxiv.get("max_results", 25)))
        for category in categories:
            params = {"search_query": f"cat:{category}", "start": 0, "max_results": max_results, "sortBy": "submittedDate", "sortOrder": "descending"}
            try:
                response = self.session.get("https://export.arxiv.org/api/query", params=params, timeout=self.timeout)
                self.record_response(response)
                response.raise_for_status()
                root = ET.fromstring(response.text)
            except (requests.RequestException, ET.ParseError) as exc:
                if "response" not in locals() or not hasattr(response, "status_code"):
                    self.record_request_error()
                self.stats["errors"] += 1
                log.warning("ArXiv query failed for %s: %s", category, exc)
                continue
            self.stats["fetched"] += 1
            for entry in root.findall(f"{{{ATOM}}}entry"):
                title = re.sub(r"\s+", " ", entry.findtext(f"{{{ATOM}}}title", "")).strip()
                summary = re.sub(r"\s+", " ", entry.findtext(f"{{{ATOM}}}summary", "")).strip()
                url = entry.findtext(f"{{{ATOM}}}id", "")
                if not url or not summary:
                    self.stats["skipped"] += 1
                    continue
                arxiv_id = url.rsplit("/", 1)[-1]
                version = arxiv_id.rsplit("v", 1)[-1] if "v" in arxiv_id and arxiv_id.rsplit("v", 1)[-1].isdigit() else "1"
                doc = Document(
                    doc_id=f"arxiv:{arxiv_id}", url=url, source="arxiv",
                    text=f"Title: {title}\n\n{summary}", title=title,
                    language="en", content_type="research", domain="arxiv.org",
                    meta={
                        "category": category,
                        "source_identity": {"type": "arxiv_paper", "arxiv_id": arxiv_id, "version": version},
                        "license_status": "unknown",
                        "rights_status": "review_required",
                    },
                )
                if self.weight_lookup:
                    doc = self._apply_weights(doc)
                if doc is not None:
                    yield doc


__all__ = ["ArxivCrawler"]
