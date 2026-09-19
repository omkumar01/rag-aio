"""SSRF-safe URL validation for the web loader and crawlers."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

from rag_core.errors import CrawlError

__all__ = ["validate_public_url"]


def _is_blocked_ip(ip: str) -> bool:
    """True when *ip* is private / loopback / link-local / reserved / unspecified."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_unspecified
        or addr.is_multicast
    )


def validate_public_url(url: str, *, allow_private: bool = False) -> str:
    """Validate *url* for outbound HTTP fetching and return a normalized URL.

    Rules:
      * Scheme must be ``http`` or ``https``.
      * The hostname must resolve and resolve only to public IPs — private, loopback,
        link-local, reserved, multicast and unspecified addresses are refused — unless
        ``allow_private=True`` (intended for localhost testing only).

    Raises :class:`~rag_core.errors.CrawlError` on any violation.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise CrawlError(
            f"URL must use http or https scheme, got {parsed.scheme!r}",
            details={"url": url},
        )
    host = parsed.hostname
    if not host:
        raise CrawlError(f"URL must include a hostname: {url!r}", details={"url": url})

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise CrawlError(f"Could not resolve hostname {host!r}: {exc}") from exc

    resolved = {str(info[4][0]) for info in infos}
    blocked = next((ip for ip in resolved if _is_blocked_ip(ip)), None)
    if blocked is not None and not allow_private:
        raise CrawlError(
            f"Refusing non-public IP {blocked!r} for host {host!r}",
            details={"host": host, "ip": blocked, "url": url},
        )
    return urlunsplit(parsed)
