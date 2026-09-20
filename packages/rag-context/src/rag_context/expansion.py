"""Neighbor/parent expansion for context assembly.

``expand_with_neighbors`` inserts supplementary chunks (siblings that a chunker
reported adjacent to a selected hit) into the candidate stream. Inserted
neighbors are annotated in metadata so they are never confused with primary
retrieval results, and already-selected ``chunk_id``s are never duplicated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rag_core.retrieval import RetrievalHit

__all__ = ["NeighborProvider", "expand_with_neighbors"]

NeighborProvider = Callable[[str], Sequence[tuple[str, str, int]] | list[tuple[str, str, int]]]


def expand_with_neighbors(
    hits: list[RetrievalHit],
    neighbor_provider: NeighborProvider,
    window: int = 1,
) -> list[RetrievalHit]:
    """Insert neighbors for each hit, up to ``window`` per hit.

    Each neighbor tuple is ``(neighbor_chunk_id, neighbor_text, neighbor_index)``
    describing the position the neighbor holds within its document. A neighbor is
    only inserted when its ``chunk_id`` is not already present and its text is
    non-empty (i.e. it "adds value"). Neighbor hits inherit the parent's score,
    strategy, and document provenance.
    """
    if window <= 0:
        return list(hits)

    existing_ids: set[str] = {h.chunk_id for h in hits}
    result: list[RetrievalHit] = []
    for hit in hits:
        result.append(hit)
        if not hit.text:
            continue
        neighbors = neighbor_provider(hit.chunk_id) or []
        taken = 0
        for neighbor_chunk_id, neighbor_text, neighbor_index in neighbors:
            if neighbor_chunk_id in existing_ids:
                continue
            if not neighbor_text or not neighbor_text.strip():
                continue
            existing_ids.add(neighbor_chunk_id)
            result.append(
                hit.model_copy(
                    update={
                        "chunk_id": neighbor_chunk_id,
                        "text": neighbor_text,
                        "score": hit.score,
                        "normalized_score": hit.normalized_score,
                        "strategy": hit.strategy,
                        "rank": hit.rank,
                        "parent_chunk_id": hit.chunk_id,
                        "metadata": {
                            **hit.metadata,
                            "is_neighbor": True,
                            "neighbor_index": neighbor_index,
                        },
                    }
                )
            )
            taken += 1
            if taken >= window:
                break
    return result
