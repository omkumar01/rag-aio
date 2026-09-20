"""Tests for normalize_query."""

from __future__ import annotations

from rag_query.normalize import normalize_query


def test_collapses_whitespace() -> None:
    assert normalize_query("hello   world\t\n  foo") == "hello world foo"


def test_strips_leading_trailing_whitespace() -> None:
    assert normalize_query("   hi there   ") == "hi there"


def test_nfkc_folding() -> None:
    # U+FB01 (fi ligature) folds to "fi" under NFKC.
    assert normalize_query("ﬁ") == "fi"
    assert normalize_query("ABC ﬁ") == "ABC fi"


def test_removes_control_chars() -> None:
    assert normalize_query("a\x00b\x07c\x1fd") == "abcd"
    # Control char between words does not fuse tokens when a space separates.
    assert normalize_query("hello\x00 world") == "hello world"


def test_optional_lowercasing() -> None:
    assert normalize_query("Hello WORLD", lowercase=True) == "hello world"
    assert normalize_query("Hello WORLD") == "Hello WORLD"


def test_empty_string() -> None:
    assert normalize_query("") == ""
    assert normalize_query("   \t  ") == ""
