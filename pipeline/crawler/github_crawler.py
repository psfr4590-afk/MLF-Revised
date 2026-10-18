"""GitHub repository metadata/content crawler."""
from __future__ import annotations

import logging
import os
from typing import Iterator
from urllib.parse import urljoin

import requests

from pipeline.crawler.base import BaseCrawler
from pipeline.types import Document

log = logging.getLogger("crawler.github")


class GitHubCrawler(BaseCrawler):
    def __init__(self, cfg: dict, weight_lookup, signal_tracker):
        super().__init__(cfg, weight_lookup, signal_tracker)
        self.base = "https://api.github.com/"
        self.timeout = int(cfg.get("timeout", 20))
        token = os.getenv(cfg.get("github", {}).get("token_env", "GITHUB_TOKEN"), "")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json", "User-Agent": "ML-Framework/1.3"})
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _request(self, path: str, params: dict | None = None):
        try:
            resp = self.session.get(urljoin(self.base, path), params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            self.record_request_error()
            self.stats["errors"] += 1
            log.warning("GitHub request failed: %s", exc)
            return None
        self.record_response(resp)
        try:
            resp.raise_for_status()
            self.stats["fetched"] += 1
            return resp.json()
        except requests.RequestException as exc:
            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                self.stats["rate_limited"] += 1
            self.stats["errors"] += 1
            log.warning("GitHub request failed: %s", exc)
            return None
            log.warning("GitHub request failed: %s", exc)
            return None

    def crawl(self) -> Iterator[Document]:
        cfg = self.cfg.get("github", {})
        topics = cfg.get("topics", [])
        languages = cfg.get("languages", [])
        min_stars = int(cfg.get("min_stars", 0))
        per_query = min(100, max(1, int(cfg.get("per_query", 10))))
        max_repos = max(0, int(cfg.get("max_repos", 0)))
        seen: set[str] = set()
        queries: list[str] = []
        for topic in topics:
            queries.append(f"topic:{topic}")
        for language in languages:
            queries.append(f"language:{language}")
        if not queries:
            queries = list(cfg.get("queries", []))
        for query in queries:
            if max_repos and len(seen) >= max_repos:
                break
            remaining = max_repos - len(seen) if max_repos else per_query
            request_size = min(per_query, remaining) if max_repos else per_query
            data = self._request(
                "search/repositories",
                {"q": query, "sort": "stars", "order": "desc", "per_page": request_size},
            ) or {}
            for repo in data.get("items", []):
                if max_repos and len(seen) >= max_repos:
                    break
                url = repo.get("html_url", "")
                if not url or url in seen or int(repo.get("stargazers_count", 0)) < min_stars:
                    continue
                seen.add(url)
                full_name = str(repo.get("full_name", repo.get("id", url)))
                default_branch = str(repo.get("default_branch", ""))
                commit = self._request(f"repos/{full_name}/commits/{default_branch}") if default_branch else None
                commit_sha = str((commit or {}).get("sha", ""))
                license_info = repo.get("license") or {}
                license_spdx = str(license_info.get("spdx_id", "")) if isinstance(license_info, dict) else ""
                text = "\n".join(filter(None, [repo.get("name", ""), repo.get("description", ""), repo.get("topics") and "Topics: " + ", ".join(repo["topics"])]))
                doc = Document(
                    doc_id=f"github:{full_name}@{commit_sha or default_branch}",
                    url=url,
                    source="github",
                    text=text,
                    title=full_name,
                    language="en",
                    content_type="documentation",
                    domain="github.com",
                    stars=int(repo.get("stargazers_count", 0)),
                    meta={
                        "full_name": full_name,
                        "default_branch": default_branch,
                        "api_url": repo.get("url", ""),
                        "source_identity": {
                            "type": "github_repository",
                            "full_name": full_name,
                            "default_branch": default_branch,
                            "commit_sha": commit_sha or None,
                        },
                        "license_status": "verified" if license_spdx and license_spdx != "NOASSERTION" else "unknown",
                        "license": license_spdx or None,
                        "rights_status": "verified" if license_spdx and license_spdx != "NOASSERTION" else "review_required",
                    },
                )
                if self.weight_lookup:
                    doc = self._apply_weights(doc)
                if doc is not None:
                    yield doc


__all__ = ["GitHubCrawler"]
