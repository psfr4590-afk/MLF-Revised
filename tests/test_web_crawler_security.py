from __future__ import annotations

import io
import socket

import pytest
import requests

from pipeline.crawler.web_crawler import WebCrawler, _pin_dns


class _DummyLookup:
    def domain_weight(self, domain):
        return 1.0, "general"

    def content_type_weight(self, content_type):
        return 1.0

    def quality_score(self, text):
        return 1.0

    def code_language_weight(self, language):
        return 1.0


class _DummySignals:
    def should_abandon(self, domain):
        return False

    def record(self, domain, score):
        pass


def _crawler(monkeypatch):
    monkeypatch.setattr("pipeline.crawler.web_crawler.time.sleep", lambda _: None)
    crawler = WebCrawler(
        {"web": {"respect_robots_txt": False, "max_redirects": 3}},
        _DummyLookup(),
        _DummySignals(),
    )
    return crawler


def test_private_ip_targets_are_rejected_without_network(monkeypatch):
    crawler = _crawler(monkeypatch)
    assert not crawler._allowed("http://127.0.0.1:8080/admin")
    assert not crawler._allowed("http://169.254.169.254/latest/meta-data/")
    assert not crawler._allowed("http://10.0.0.7/internal")


def test_hostname_with_private_resolution_is_rejected(monkeypatch):
    crawler = _crawler(monkeypatch)
    monkeypatch.setattr(
        "pipeline.crawler.web_crawler.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("192.168.1.20", 0))],
    )
    assert not crawler._allowed("https://example.test/")


def test_redirect_to_private_address_is_blocked(monkeypatch):
    crawler = _crawler(monkeypatch)
    monkeypatch.setattr(
        "pipeline.crawler.web_crawler.WebCrawler._host_is_public",
        staticmethod(lambda host: host not in {"127.0.0.1", "localhost"}),
    )
    monkeypatch.setattr(crawler, "_validated_addresses", lambda url: ("93.184.216.34",))

    redirect = requests.Response()
    redirect.status_code = 302
    redirect.headers["Location"] = "http://127.0.0.1:8080/internal"
    redirect.url = "https://public.example/"
    redirect._content = b""
    redirect.raw = io.BytesIO()

    calls = []

    class FakeSession:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return redirect

    crawler.session = FakeSession()
    assert crawler._fetch("https://public.example/") is None
    assert calls == [("https://public.example/", {"timeout": 15.0, "allow_redirects": False})]


def test_public_redirect_is_followed_manually(monkeypatch):
    crawler = _crawler(monkeypatch)
    monkeypatch.setattr(
        "pipeline.crawler.web_crawler.WebCrawler._host_is_public",
        staticmethod(lambda host: True),
    )
    monkeypatch.setattr(crawler, "_validated_addresses", lambda url: ("93.184.216.34",))

    first = requests.Response()
    first.status_code = 302
    first.headers["Location"] = "/next"
    first.url = "https://public.example/"
    first._content = b""
    first.raw = io.BytesIO()

    second = requests.Response()
    second.status_code = 200
    second.url = "https://public.example/next"
    second._content = b"<html><title>ok</title>body</html>"

    responses = iter([first, second])
    calls = []

    class FakeSession:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return next(responses)

    crawler.session = FakeSession()
    result = crawler._fetch("https://public.example/")
    assert result is second
    assert [url for url, _ in calls] == ["https://public.example/", "https://public.example/next"]
    assert all(kwargs["allow_redirects"] is False for _, kwargs in calls)


def test_robots_fetch_failure_fails_closed(monkeypatch):
    crawler = WebCrawler(
        {"web": {"respect_robots_txt": True}},
        _DummyLookup(),
        _DummySignals(),
    )
    monkeypatch.setattr(crawler, "_validated_addresses", lambda url: ("93.184.216.34",))

    def fail_get(*args, **kwargs):
        raise requests.RequestException("network unavailable")

    monkeypatch.setattr(crawler.session, "get", fail_get)
    assert not crawler._robots_allowed("https://public.example/")


def test_oversized_response_is_not_materialized(monkeypatch):
    crawler = _crawler(monkeypatch)
    crawler.web["seed_urls"] = ["https://public.example/"]
    monkeypatch.setattr(
        "pipeline.crawler.web_crawler.WebCrawler._host_is_public",
        staticmethod(lambda host: True),
    )
    monkeypatch.setattr(crawler, "_validated_addresses", lambda url: ("93.184.216.34",))

    response = requests.Response()
    response.status_code = 200
    response.url = "https://public.example/"
    response.headers["Content-Type"] = "text/html"
    response.iter_content = lambda chunk_size: iter([b"a" * 8, b"b" * 8])
    response.raw = io.BytesIO()

    class FakeSession:
        def get(self, url, **kwargs):
            return response

    crawler.session = FakeSession()
    crawler.web["max_content_bytes"] = 10
    assert list(crawler.crawl()) == []
    assert crawler.stats["skipped"] == 1
    assert crawler.stats["fetched"] == 1


def test_pin_dns_replaces_hostname_resolution_with_validated_addresses(monkeypatch):
    calls = []

    def fake_getaddrinfo(node, port, family=0, type=0, proto=0, flags=0):
        calls.append(node)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port or 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with _pin_dns("example.com", ("93.184.216.34",)):
        result = socket.getaddrinfo("example.com", 443, type=socket.SOCK_STREAM)
    assert result[0][4][0] == "93.184.216.34"
    assert calls == ["93.184.216.34"]


def test_fetch_pins_the_actual_connection_to_the_validated_address(monkeypatch):
    crawler = _crawler(monkeypatch)
    observed = []

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "text/plain"}
        url = "https://example.com/"
        is_redirect = False
        is_permanent_redirect = False

        def raise_for_status(self):
            pass

        def close(self):
            pass

    def fake_get(url, **kwargs):
        observed.append(socket.getaddrinfo("example.com", 443, type=socket.SOCK_STREAM)[0][4][0])
        return FakeResponse()

    monkeypatch.setattr(crawler, "_allowed", lambda url: True)
    monkeypatch.setattr(crawler, "_robots_allowed", lambda url: True)
    monkeypatch.setattr(crawler, "_validated_addresses", lambda url: ("93.184.216.34",))
    monkeypatch.setattr(crawler.session, "get", fake_get)
    monkeypatch.setattr(crawler, "_polite_wait", lambda domain: None)

    response = crawler._fetch("https://example.com/")
    assert response is not None
    assert observed == ["93.184.216.34"]


def test_validated_addresses_reject_private_or_local_resolution(monkeypatch):
    crawler = _crawler(monkeypatch)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))
        ],
    )
    assert crawler._validated_addresses("https://example.com/") is None


def test_pinned_dns_context_restores_socket_resolution_on_exception():
    original = socket.getaddrinfo
    with pytest.raises(RuntimeError):
        with _pin_dns("example.com", ("93.184.216.34",)):
            raise RuntimeError("test")
    assert socket.getaddrinfo is original


def test_crawler_disables_environment_proxies():
    crawler = WebCrawler(
        {"web": {"respect_robots_txt": False}},
        _DummyLookup(),
        _DummySignals(),
    )
    assert crawler.session.trust_env is False
