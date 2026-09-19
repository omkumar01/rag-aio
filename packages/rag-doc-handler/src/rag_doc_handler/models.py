"""Loaded-source model for raw bytes + metadata discovered before parsing."""

from __future__ import annotations

from pydantic import Field
from rag_core.base import RagBaseModel


class LoadedSource(RagBaseModel):
    """Raw bytes and ambient metadata produced by a :class:`~rag_doc_handler.loaders` loader.

    ``source_uri`` is the normalized origin (resolved file path or final URL), ``data`` is
    the raw payload, ``mime_type`` is optional and derived from sniffing/headers, and
    ``metadata`` carries any loader-specific keys (e.g. final URL, content-type).
    """

    source_uri: str
    data: bytes
    mime_type: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
