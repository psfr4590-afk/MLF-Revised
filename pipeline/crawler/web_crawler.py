"""General-purpose web crawler used by Model Lab.

The crawler is deliberately polite and bounded: it honors robots.txt when configured,
uses request timeouts/retries, stays within configured depth/page budgets, applies
source-quality gating, and emits canonical :class:`pipeline.types.Document` records.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from pipeline.crawler.base import BaseCrawler
from pipeline.crawler.content_parser import ContentParseError, parse_content
from pipeline.crawler.source_scorer import classify_text, classify_url, detect_code_language, score_document
from pipeline.types import Document

log = logging.getLogger("crawler.web")
_DNS_PIN_LOCK = threading.RLock()


@contextmanager
def _pin_dns(hostname: str, addresses: tuple[str, ...]):
    """Pin socket DNS resolution for *hostname* to an already validated address set."""
    host = hostname.rstrip(".").lower()
    original_getaddrinfo = socket.getaddrinfo

    def pinned_getaddrinfo(node, port, family=0, type=0, proto=0, flags=0):
        if isinstance(node, str) and node.rstrip(".").lower() == host:
            results = []
            for address in addresses:
                results.extend(original_getaddrinfo(address, port, family, type, proto, flags))
            if not results:
                raise socket.gaierror(socket.EAI_NONAME, "Pinned DNS address family unavailable")
            return results
        return original_getaddrinfo(node, port, family, type, proto, flags)

    with _DNS_PIN_LOCK:
        socket.getaddrinfo = pinned_getaddrinfo
        try:
            yield
        finally:
            socket.getaddrinfo = original_getaddrinfo


class WebCrawler(BaseCrawler):
    def __init__(self, cfg: dict, weight_lookup, signal_tracker):
        super().__init__(cfg, weight_lookup, signal_tracker)
        self.web = cfg.get("web", {})
        self.max_depth = int(self.web.get("max_depth", 2))
        self.max_pages_per_domain = int(self.web.get("max_pages_per_domain", 500))
        self.timeout = float(cfg.get("request_timeout", self.web.get("request_timeout", 15)))
        self.retries = int(cfg.get("retry_attempts", 3))
        self.backoff = float(cfg.get("retry_backoff", 2.0))
        self.delay = float(cfg.get("politeness_delay", 1.0))
        self.follow_links = bool(self.web.get("follow_links", True))
        self.stay_on_domain = bool(self.web.get("stay_on_domain", False))
        self.respect_robots = bool(self.web.get("respect_robots_txt", True))
        self.allow = [re.compile(p) for p in self.web.get("url_allowlist_patterns", []) if p]
        self.block = [re.compile(p) for p in self.web.get("url_blocklist_patterns", []) if p]
        self.user_agent = str(cfg.get("user_agent", "ModelLab/1.3 (+local research dataset builder)"))
        self.max_redirects = max(0, int(self.web.get("max_redirects", 5)))
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/pdf,text/csv,text/tab-separated-values,application/json,application/xml,text/xml,text/plain,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation",
        })
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}
        self._seen_urls: set[str] = set()
        self._domain_pages: dict[str, int] = {}

    @staticmethod
    def _resolve_addresses(hostname: str | None) -> tuple[str, ...]:
        if not hostname:
            return ()
        host = hostname.rstrip(".")
        try:
            literal = ipaddress.ip_address(host)
            return (str(literal),)
        except ValueError:
            try:
                addresses = {
                    ipaddress.ip_address(info[4][0])
                    for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
                }
            except (OSError, ValueError):
                return ()
        return tuple(sorted(str(address) for address in addresses))

    @classmethod
    def _host_is_public(cls, hostname: str | None) -> bool:
        """Return True only when every resolved address is globally routable."""
        addresses = cls._resolve_addresses(hostname)
        return bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)

    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        if not self._host_is_public(parsed.hostname):
            return False
        if self.block and any(p.search(url) for p in self.block):
            return False
        return not self.allow or any(p.search(url) for p in self.allow)

    def _validated_addresses(self, url: str) -> tuple[str, ...] | None:
        """Resolve and validate the exact address set that the next request will use."""
        hostname = urlparse(url).hostname
        addresses = self._resolve_addresses(hostname)
        if not addresses or not all(ipaddress.ip_address(address).is_global for address in addresses):
            return None
        return addresses

    def _pinned_get(self, url: str, addresses: tuple[str, ...]):
        with _pin_dns(urlparse(url).hostname or "", addresses):
            return self.session.get(url, timeout=self.timeout, allow_redirects=False)

    def _robots_allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots.get(root)
        if rp is None:
            robots_url = urljoin(root, "/robots.txt")
            addresses = self._validated_addresses(robots_url)
            if addresses is None:
                self._robots[root] = RobotFileParser()
                return False
            rp = RobotFileParser(robots_url)
            try:
                response = self._pinned_get(robots_url, addresses)
                if 200 <= response.status_code < 300:
                    rp.parse(response.content.decode("utf-8", errors="replace").splitlines())
                elif response.status_code in (401, 403):
                    rp.disallow_all = True
                elif 400 <= response.status_code < 500:
                    rp.allow_all = True
                else:
                    log.warning("robots.txt unavailable for %s: HTTP %s", root, response.status_code)
                    response.close()
                    self._robots[root] = RobotFileParser()
                    return False
                response.close()
            except requests.RequestException as exc:
                log.warning("robots.txt unavailable for %s: %s", root, exc)
                self._robots[root] = RobotFileParser()
                return False
            self._robots[root] = rp
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception as exc:
            log.warning("robots policy could not be evaluated for %s: %s", root, exc)
            return False

    def _polite_wait(self, domain: str) -> None:
        now = time.monotonic()
        delta = now - self._last_request.get(domain, 0.0)
        if delta < self.delay:
            time.sleep(self.delay - delta)
        self._last_request[domain] = time.monotonic()

    def _fetch(self, url: str):
        current_url = url
        last = None
        for redirect in range(self.max_redirects + 1):
            parsed = urlparse(current_url)
            if not self._allowed(current_url) or not self._robots_allowed(current_url):
                self.stats["skipped"] += 1
                return None
            addresses = self._validated_addresses(current_url)
            if addresses is None:
                self.stats["skipped"] += 1
                return None
            domain = parsed.netloc.lower()
            self._polite_wait(domain)
            for attempt in range(self.retries + 1):
                response = None
                try:
                    response = self._pinned_get(current_url, addresses)
                    self.record_response(response)
                    if response.is_redirect or response.is_permanent_redirect:
                        location = response.headers.get("Location")
                        response.close()
                        if not location or redirect >= self.max_redirects:
                            self.stats["skipped"] += 1
                            return None
                        current_url = urljoin(current_url, location).split("#", 1)[0]
                        if not self._allowed(current_url):
                            self.stats["skipped"] += 1
                            return None
                        break
                    if response.status_code == 429:
                        self.record_retry()
                        retry_after = response.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff * (attempt + 1)
                        response.close()
                        time.sleep(min(wait, 60.0))
                        continue
                    response.raise_for_status()
                    self.stats["fetched"] += 1
                    return response
                except requests.RequestException as exc:
                    last = exc
                    self.stats["errors"] += 1
                    if "response" not in locals() or response is None:
                        self.record_request_error()
                    if attempt < self.retries:
                        self.record_retry()
                        time.sleep(self.backoff * (attempt + 1))
            else:
                break
            continue
        if last:
            log.warning("fetch failed %s: %s", url, last)
        return None

    def _document(self, url: str, title: str, text: str, depth: int, content_type: str, metadata: dict | None = None) -> Document:
        parsed = urlparse(url)
        semantic_type = classify_url(url)
        if semantic_type == "unknown":
            semantic_type = content_type or classify_text(text)
        doc = Document(
            doc_id=hashlib.sha256(url.encode("utf-8")).hexdigest(), url=url, source="web", text=text,
            title=title, content_type=semantic_type, code_language=detect_code_language(url),
            domain=parsed.netloc.lower(), crawl_depth=depth,
        )
        if doc.content_type == "unknown":
            doc.content_type = classify_text(text)
        if metadata:
            doc.meta.update({"parser": metadata})
        doc.meta["quality_signal"] = score_document(url, text)
        self.signal_tracker.record(doc.domain, doc.meta["quality_signal"])
        return self._apply_weights(doc)

    def crawl(self):
        seeds = list(self.web.get("seed_urls", []))
        queue = deque((u, 0, u) for u in seeds if self._allowed(u))
        while queue:
            url, depth, origin = queue.popleft()
            normalized = url.split("#", 1)[0]
            if normalized in self._seen_urls or not self._allowed(normalized):
                continue
            domain = urlparse(normalized).netloc.lower()
            if self._domain_pages.get(domain, 0) >= self.max_pages_per_domain:
                self.stats["skipped"] += 1
                continue
            if self.signal_tracker.should_abandon(domain):
                self.stats["abandoned_domains"] += 1
                continue
            if not self._robots_allowed(normalized):
                self.stats["skipped"] += 1
                continue
            self._seen_urls.add(normalized)
            response = self._fetch(normalized)
            if response is None:
                continue
            final_url = response.url.split("#", 1)[0]
            if not self._allowed(final_url):
                response.close()
                self.stats["skipped"] += 1
                continue
            response_content_type = response.headers.get("Content-Type", "").lower()
            max_bytes = int(self.web.get("max_content_bytes", 5_000_000))
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                response.close()
                self.stats["skipped"] += 1
                continue
            try:
                chunks = []
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        self.stats["skipped"] += 1
                        break
                    chunks.append(chunk)
                else:
                    body = b"".join(chunks)
                    self._domain_pages[domain] = self._domain_pages.get(domain, 0) + 1
                    try:
                        parsed_content = parse_content(final_url, response_content_type, body)
                    except ContentParseError as exc:
                        self.stats["skipped"] += 1
                        log.warning("content parsing failed for %s: %s", final_url, exc)
                        continue
                    links = [urljoin(final_url, link) for link in parsed_content.links]
                    doc = self._document(
                        final_url,
                        parsed_content.title,
                        parsed_content.text,
                        depth,
                        parsed_content.content_type,
                        parsed_content.metadata,
                    )
                    yield doc
                    if not self.follow_links or depth >= self.max_depth:
                        continue
                    for link in links:
                        link = link.split("#", 1)[0]
                        if not self._allowed(link):
                            continue
                        if self.stay_on_domain and urlparse(link).netloc.lower() != urlparse(origin).netloc.lower():
                            continue
                        if link not in self._seen_urls:
                            queue.append((link, depth + 1, origin))
            finally:
                response.close()
