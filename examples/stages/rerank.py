"""Stage 3 — Reranking: reorder retrieval candidates with the heuristic reranker.

:class:`rag_rerank.HeuristicReranker` is the zero-dependency Jaccard baseline
that the local pipeline uses by default. This example feeds it a deliberately
mis-ordered candidate list and prints the before/after ordering, including the
original rank/score preserved for explainability via :func:`to_rerank_hits`.

Fully offline; no models involved.

Run: uv run python examples/stages/rerank.py
"""

from __future__ import annotations

import asyncio

from rag_core import Query, RetrievalHit
from rag_rerank import HeuristicReranker, RerankConfig, to_rerank_hits

QUERY_TEXT = "how often should the coolant pumps be inspected"

# Candidates deliberately in a bad order: the correct answer sits last.
CANDIDATE_TEXTS = [
    "Replace air filters quarterly and log the change in the journal.",
    "The facility cafeteria opens at eleven; the coffee machine is reliable.",
    "Inspect coolant pumps every ninety days; record findings in the log.",
]


def _candidate(hit_text: str, rank: int) -> RetrievalHit:
    """Build a retrieval hit with a descending-but-meaningless placeholder score."""
    return RetrievalHit(
        chunk_id=f"chunk-{rank}",
        document_id="manual.md",
        score=1.0 - rank * 0.1,
        normalized_score=1.0 - rank * 0.1,
        rank=rank,
        strategy="dense",
        text=hit_text,
    )


async def main() -> None:
    candidates = [_candidate(text, rank) for rank, text in enumerate(CANDIDATE_TEXTS)]

    print(f"Query: {QUERY_TEXT!r}")
    print("\nBefore reranking (retrieval order):")
    for hit in candidates:
        print(f"  #{hit.rank} score={hit.score:.2f}  {hit.text!r}")

    reranker = HeuristicReranker(RerankConfig(top_k=3))
    reranked = await reranker.rerank(Query(text=QUERY_TEXT), candidates)

    print("\nAfter reranking (heuristic Jaccard term overlap):")
    for audit in to_rerank_hits(candidates, reranked):
        moved = f"was #{audit.original_rank}" if audit.original_rank != audit.rank else "unchanged"
        print(f"  #{audit.rank} score={audit.score:.3f} ({moved})  {audit.text!r}")


if __name__ == "__main__":
    asyncio.run(main())
