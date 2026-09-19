"""Explainability helpers for retrieval results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rag_core.retrieval import RetrievalResult


def explain_result(result: RetrievalResult) -> dict[str, Any]:
    """Produce a JSON-serializable per-hit explanation of a retrieval result."""
    return {
        "query_id": result.query_id,
        "strategies": list(result.strategies),
        "timings_ms": dict(result.timings_ms),
        "total_candidates": result.total_candidates,
        "hits": [
            {
                "rank": h.rank,
                "chunk_id": h.chunk_id,
                "document_id": h.document_id,
                "strategy": h.strategy,
                "model": h.model,
                "score": h.score,
                "normalized_score": h.normalized_score,
                "strategy_scores": dict(h.strategy_scores),
                "filters_applied": dict(h.filters_applied),
            }
            for h in result.hits
        ],
    }


def attach_text(
    result: RetrievalResult,
    text_lookup: Callable[[str], str | None] | Mapping[str, str | None] | None,
) -> RetrievalResult:
    """Fill ``hit.text`` from a chunk provider (callable or mapping).

    Only hits without text are populated; existing text is preserved.
    """
    if text_lookup is None:
        return result
    if isinstance(text_lookup, Mapping):
        for hit in result.hits:
            if hit.text is None:
                hit.text = text_lookup.get(hit.chunk_id)
    else:
        for hit in result.hits:
            if hit.text is None:
                found = text_lookup(hit.chunk_id)
                if isinstance(found, str):
                    hit.text = found
    return result


__all__ = ["attach_text", "explain_result"]
