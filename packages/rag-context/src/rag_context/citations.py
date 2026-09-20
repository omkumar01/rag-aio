"""Citation construction from an assembled :class:`~rag_core.context.Context`.

A citation is emitted for every context item that carries a ``citation_id``, in
final rendering order. A caller-provided ``source_lookup`` can backfill missing
``source_uri`` / ``page_numbers`` keyed by ``chunk_id``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rag_core.context import Citation, Context

__all__ = ["SourceLookup", "build_citations"]

SourceLookup = Callable[[str], tuple[str, Sequence[int]] | None]


def build_citations(
    context: Context,
    source_lookup: SourceLookup | None = None,
) -> list[Citation]:
    """Build one :class:`~rag_core.context.Citation` per cited item, in order."""
    citations: list[Citation] = []
    for item in context.items:
        citation_id = item.citation_id
        if citation_id is None:
            continue
        source_uri = item.source_uri
        page_numbers: list[int] = list(item.page_numbers)
        if source_lookup is not None and (source_uri is None or not page_numbers):
            looked_up = source_lookup(item.chunk_id)
            if looked_up is not None:
                source_uri_part, page_numbers_part = looked_up
                if source_uri is None and source_uri_part:
                    source_uri = source_uri_part
                if not page_numbers and page_numbers_part:
                    page_numbers = list(page_numbers_part)
        quote = _quote(item.text)
        citations.append(
            Citation(
                citation_id=citation_id,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                source_uri=source_uri,
                page_numbers=page_numbers,
                quote=quote,
            )
        )
    return citations


def _quote(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped[:120]
