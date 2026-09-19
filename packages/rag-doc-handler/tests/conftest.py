"""Shared test fixtures for rag-doc-handler."""

from __future__ import annotations

import ipaddress
import socket

import pytest


def _public_addrinfo(host: str, *args: object, **kwargs: object):
    """getaddrinfo replacement: IP literals resolve to themselves, domains to a public IP.

    This keeps every test offline (no real DNS) while still allowing IP-literal
    addresses (e.g. ``127.0.0.1``) to be classified correctly by SSRF validation.
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", 0))]
    if addr.version == 4:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (str(addr), 0))]
    return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (str(addr), 0, 0, 0))]


@pytest.fixture(autouse=True)
def _offline_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make public URL validation offline: mock DNS, never hit the network."""
    monkeypatch.setattr("rag_doc_handler.security.socket.getaddrinfo", _public_addrinfo)
