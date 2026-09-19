"""Serialization helpers for documents stored in SQL backends.

The canonical document is serialized in full as compact JSON in a ``data``
TEXT column (the single source of truth for round-tripping). The remaining
columns are *query* columns populated from the same document so that common
lookups (by hash, source, namespace) do not require decoding the JSON blob.
"""

from __future__ import annotations

import json
from typing import Any

from rag_core.documents import Document
from rag_core.serde import from_json


def document_to_row_fields(doc: Document) -> dict[str, Any]:
    """Project a :class:`Document` into the column values of the ``documents`` table.

    ``data`` holds the full JSON serialization (lossless); every other column is
    a denormalized query hint derived from the same document.
    """
    md = doc.metadata
    return {
        "id": doc.id,
        "content_hash": doc.content_hash,
        "source_uri": doc.source_uri,
        "tenant": doc.tenant or "",
        "namespace": doc.namespace or "",
        "title": md.title or "",
        "mime_type": md.mime_type or "",
        "text": doc.text,
        "metadata_json": md.model_dump_json(),
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "data": doc.model_dump_json(),
    }


def row_to_document(fields: dict[str, Any]) -> Document:
    """Rebuild a :class:`Document` from a row's full-serialization ``data`` column.

    Uses :func:`rag_core.serde.from_json` so validation matches the canonical
    boundary object schema exactly.
    """
    raw = fields["data"]
    if isinstance(raw, str):
        return from_json(raw, Document)
    # Defensive fallback for row factories returning native types.
    return from_json(json.dumps(raw), Document)


__all__ = ["document_to_row_fields", "row_to_document"]
