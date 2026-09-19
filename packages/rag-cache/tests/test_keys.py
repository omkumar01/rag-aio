"""Tests for cache key construction."""

from __future__ import annotations

from rag_cache import cache_key


def test_cache_key_format() -> None:
    key = cache_key("ns", foo="bar")
    assert key.startswith("ns:")
    ns, _, digest = key.partition(":")
    assert ns == "ns"
    assert len(digest) == 64
    int(digest, 16)  # must be valid hex


def test_cache_key_deterministic() -> None:
    assert cache_key("ns", foo="bar") == cache_key("ns", foo="bar")


def test_cache_key_order_insensitive_for_dict_parts() -> None:
    assert cache_key("ns", a=1, b=2) == cache_key("ns", b=2, a=1)


def test_cache_key_different_parts_differ() -> None:
    assert cache_key("ns", a=1) != cache_key("ns", a=2)


def test_cache_key_different_namespace_differs() -> None:
    assert cache_key("ns1", a=1) != cache_key("ns2", a=1)


def test_cache_key_no_parts_is_stable() -> None:
    assert cache_key("ns") == cache_key("ns")
