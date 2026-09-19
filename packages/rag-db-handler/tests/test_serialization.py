"""Tests for document serialization helpers (lossless round-trip)."""

from __future__ import annotations

import pytest
from rag_core.documents import Document, DocumentMetadata
from rag_core.serde import from_json
from rag_db_handler.serialization import document_to_row_fields, row_to_document


def _doc() -> Document:
    return Document(
        source_uri="mem://doc/1",
        text="hello world\nsecond line",
        metadata=DocumentMetadata(
            title="My Title",
            mime_type="text/plain",
            author="tester",
            language="en",
            custom={"foo": "bar"},
        ),
        tenant="t1",
        namespace="ns1",
        pages=[],
    )


@pytest.mark.unit
def test_document_to_row_fields_has_query_columns() -> None:
    doc = _doc()
    fields = document_to_row_fields(doc)
    for key in (
        "id",
        "content_hash",
        "source_uri",
        "tenant",
        "namespace",
        "title",
        "mime_type",
        "text",
        "metadata_json",
        "created_at",
        "updated_at",
        "data",
    ):
        assert key in fields
    assert fields["id"] == doc.id
    assert fields["source_uri"] == doc.source_uri
    assert fields["tenant"] == "t1"
    assert fields["namespace"] == "ns1"
    assert fields["title"] == "My Title"
    assert fields["mime_type"] == "text/plain"
    assert fields["text"] == doc.text
    assert "data" in fields


@pytest.mark.unit
def test_row_to_document_roundtrip_lossless() -> None:
    doc = _doc()
    fields = document_to_row_fields(doc)
    rebuilt = row_to_document(fields)
    assert rebuilt.id == doc.id
    assert rebuilt.source_uri == doc.source_uri
    assert rebuilt.text == doc.text
    assert rebuilt.content_hash == doc.content_hash
    assert rebuilt.metadata.title == "My Title"
    assert rebuilt.metadata.mime_type == "text/plain"
    assert rebuilt.metadata.author == "tester"
    assert rebuilt.metadata.language == "en"
    assert rebuilt.metadata.custom == {"foo": "bar"}
    assert rebuilt.tenant == "t1"
    assert rebuilt.namespace == "ns1"
    # full JSON round-trips exactly via the data column
    assert rebuilt.model_dump_json() == doc.model_dump_json()
    # and the canonical deserializer agrees
    assert from_json(doc.model_dump_json(), Document).id == from_json(fields["data"], Document).id
