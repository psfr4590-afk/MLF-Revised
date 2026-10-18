"""Optional Google Programmable Search crawler."""
from __future__ import annotations

import logging
import os
from typing import Iterator

import requests

from pipeline.crawler.base import BaseCrawler
from pipeline.types import Document

log = logging.getLogger("crawler.google")


class GoogleCrawler(BaseCrawler):
    def __init__(self, cfg: dict, weight_lookup, signal_tracker):
        super().__init__(cfg, weight_lookup, signal_tracker)
        self.google = cfg.get("google", {})
        self.timeout = int(cfg.get("timeout", 20))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ML-Framework/1.3"

    def crawl(self) -> Iterator[Document]:
        api_key = os.getenv(self.google.get("api_key_env", "GOOGLE_API_KEY"), "")
        cx = os.getenv(self.google.get("cx_env", "GOOGLE_CX"), "")
        queries = self.google.get("queries", [])
        if not api_key or not cx:
            log.info("Google crawler disabled: GOOGLE_API_KEY/GOOGLE_CX not configured")
            return
        for query in queries:
            params = {"key": api_key, "cx": cx, "q": query, "num": min(10, int(self.google.get("num", 10)))}
            try:
                resp = self.session.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=self.timeout)
                self.record_response(resp)
                resp.raise_for_status()
                payload = resp.json()
                self.stats["fetched"] += 1
            except (requests.RequestException, ValueError) as exc:
                if "resp" not in locals() or not hasattr(resp, "status_code"):
                    self.record_request_error()
                self.stats["errors"] += 1
                log.warning("Google search failed for %r: %s", query, exc)
                continue
            for item in payload.get("items", []):
                url = item.get("link", "")
                text = "\n".join(filter(None, [item.get("title", ""), item.get("snippet", "")]))
                if not url or not text:
                    continue
                doc = Document(
                    doc_id=f"google:{url}", url=url, source="google", text=text,
                    title=item.get("title", ""), language="en", content_type="documentation",
                    domain="google.com",
                    meta={
                        "query": query,
                        "displayLink": item.get("displayLink", ""),
                        "source_identity": {"type": "search_result", "result_url": url, "query": query},
                        "license_status": "unknown",
                        "rights_status": "review_required",
                    },
                )
                if self.weight_lookup:
                    doc = self._apply_weights(doc)
                if doc is not None:
                    yield doc


__all__ = ["GoogleCrawler"]
