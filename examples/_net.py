"""Shared network guard for the examples.

The examples take an endpoint URL from the environment and then make HTTP
requests to it. Because that URL is user-controlled input flowing into
outbound requests, every example validates it first: only ``http``/``https``
schemes are accepted, and by default only loopback hosts (the local-first
default for rag-aio) are reachable. Pointing an example at a public endpoint
requires an explicit opt-in via ``RAG_ALLOW_PUBLIC_ENDPOINT=1``.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse


class EndpointBlocked(ValueError):
    """Raised when an endpoint URL is not allowed by the network guard."""


def validate_base_url(base_url: str) -> str:
    """Return *base_url* if it passes the example network guard.

    Raises :class:`EndpointBlocked` otherwise. See the module docstring for
    the policy.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise EndpointBlocked(
            f"only http/https endpoints are allowed, got scheme {parsed.scheme!r}"
        )
    host = parsed.hostname
    if not host:
        raise EndpointBlocked(f"endpoint URL has no host: {base_url!r}")
    if _is_loopback(host):
        return base_url
    if os.environ.get("RAG_ALLOW_PUBLIC_ENDPOINT", "").lower() in ("1", "true", "yes"):
        return base_url
    raise EndpointBlocked(
        f"refusing non-loopback endpoint {host!r}; set RAG_ALLOW_PUBLIC_ENDPOINT=1 to allow it"
    )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Resolve names so e.g. a DNS name pointing at a loopback address is
        # still allowed, while public names fall through to the opt-in check.
        import socket

        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        return bool(infos) and all(ipaddress.ip_address(info[4][0]).is_loopback for info in infos)
