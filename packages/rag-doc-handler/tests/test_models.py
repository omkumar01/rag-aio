"""Tests for rag_doc_handler.models."""

from __future__ import annotations

from rag_doc_handler.models import LoadedSource


def test_loaded_source_required_fields() -> None:
    src = LoadedSource(source_uri="file:///a.txt", data=b"hi", mime_type="text/plain")
    assert src.source_uri == "file:///a.txt"
    assert src.data == b"hi"
    assert src.mime_type == "text/plain"
    assert src.metadata == {}


def test_loaded_source_defaults() -> None:
    src = LoadedSource(source_uri="http://x/y", data=b"")
    assert src.mime_type is None
    assert src.metadata == {}


def test_loaded_source_metadata() -> None:
    src = LoadedSource(source_uri="x", data=b"", metadata={"final_url": "http://x/y"})
    assert src.metadata["final_url"] == "http://x/y"


def test_loaded_source_forbids_extra_fields() -> None:
    from pydantic import ValidationError

    try:
        LoadedSource(source_uri="x", data=b"", unexpected="no")  # type: ignore[call-arg]
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for extra field")
