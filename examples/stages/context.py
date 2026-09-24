"""Stage 4 — Context engineering: budget hits into a prompt-ready context.

:class:`rag_context.ContextBuilderImpl` takes retrieval hits and produces a
:class:`rag_core.Context` honoring a token budget, dedup, ordering strategy,
and numeric citation ids. The rendered text (``render_context``) and the
citation records (``build_citations``) are exactly what a generation stage
would consume.

Fully offline; no models involved.

Run: uv run python examples/stages/context.py
"""

from __future__ import annotations

import asyncio

from rag_context import (
    ContextBuilderImpl,
    ContextConfig,
    ContextTokenizer,
    WhitespaceCounter,
    build_citations,
    render_context,
)
from rag_core import Query, RetrievalHit

QUERY_TEXT = "what is the escalation path for retrieval incidents"

HITS = [
    RetrievalHit(
        chunk_id="runbook-1",
        document_id="oncall.md",
        score=0.91,
        normalized_score=0.91,
        rank=0,
        strategy="hybrid",
        text="Retrieval incidents page the relevance engineer first. "
        "If recall is affected, escalate to the platform lead within 30 minutes.",
        metadata={"source_uri": "docs/oncall.md", "document_title": "On-call Runbook"},
    ),
    RetrievalHit(
        chunk_id="runbook-2",
        document_id="oncall.md",
        score=0.74,
        normalized_score=0.74,
        rank=1,
        strategy="hybrid",
        text="Sev1 incidents require a status page update within 15 minutes. "
        "The incident commander role rotates weekly.",
        metadata={"source_uri": "docs/oncall.md", "document_title": "On-call Runbook"},
    ),
    RetrievalHit(
        chunk_id="changelog-9",
        document_id="changelog.md",
        score=0.58,
        normalized_score=0.58,
        rank=2,
        strategy="hybrid",
        text="The cafeteria menu changed in March; this entry is irrelevant filler "
        "that the token budget may drop.",
        metadata={"source_uri": "docs/changelog.md", "document_title": "Changelog"},
    ),
]


async def main() -> None:
    builder = ContextBuilderImpl(
        ContextTokenizer(WhitespaceCounter()),
        ContextConfig(token_budget=120, reserve_for_answer=32),
    )
    context = await builder.build(Query(text=QUERY_TEXT), HITS, token_budget=120)

    print(f"Query: {QUERY_TEXT!r}")
    print(f"Strategy: {context.strategy} | truncated: {context.truncated}")
    print(f"Selected {len(context.items)} of {len(HITS)} hits\n")

    print("Rendered context:")
    print(render_context(context))

    print("\nCitations:")
    for citation in build_citations(context):
        print(
            f"  [{citation.citation_id}] {citation.source_uri} "
            f"(chunk {citation.chunk_id}): {citation.quote}"
        )


if __name__ == "__main__":
    asyncio.run(main())
