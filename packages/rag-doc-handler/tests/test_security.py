"""Tests for rag_doc_handler.security (SSRF validation). Offline: DNS is mocked in conftest."""

from __future__ import annotations

import pytest
from rag_core.errors import CrawlError
from rag_doc_handler.security import validate_public_url


def test_rejects_non_http_scheme() -> None:
    with pytest.raises(CrawlError, match="scheme"):
        validate_public_url("ftp://example.com")


@pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.1", "10.0.0.5", "169.254.1.1"])
def test_rejects_private_ips(host: str) -> None:
    with pytest.raises(CrawlError, match="non-public"):
        validate_public_url(f"http://{host}/path")


def test_rejects_no_hostname() -> None:
    with pytest.raises(CrawlError, match="hostname"):
        validate_public_url("http://")


def test_accepts_public_url_when_dns_mocked() -> None:
    result = validate_public_url("http://example.com/a?b=c#d")
    assert result.startswith("http://example.com")


def test_allows_private_when_allowed() -> None:
    # allow_private=True bypasses the private-IP block (localhost testing).
    result = validate_public_url("http://127.0.0.1/", allow_private=True)
    assert result == "http://127.0.0.1/"


def test_rejects_ipv6_loopback() -> None:
    with pytest.raises(CrawlError, match="non-public"):
        validate_public_url("http://[::1]/x")


def test_rejects_private_when_not_allowed_ipv6() -> None:
    # fc00::/7 (unique local) is blocked by default.
    with pytest.raises(CrawlError, match="non-public"):
        validate_public_url("http://[fd12:3456:789a::1]/x")
