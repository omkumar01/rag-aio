"""Stage 2 — Retrieval: dense and hybrid (dense + BM25) over an in-memory index.

Builds a real index with the rag-embedder pipeline (chunk -> embed -> upsert)
into :class:`rag_db_handler.InMemoryVectorStore`, then runs:

* :class:`rag_retrieval.DenseRetriever` — cosine similarity over the store;
* a :class:`rag_retrieval.HybridRetriever` fusing dense + :class:`BM25Retriever`
  with :class:`ReciprocalRankFusion`.

Embeddings use the deterministic hash-based :class:`MockEmbedder`, so scores
are stable across runs but not semantically meaningful — this example shows the
mechanics, not relevance quality. Fully offline.

Run: uv run python examples/stages/retrieve.py
"""

from __future__ import annotations

import asyncio

from rag_core import Document, Query
from rag_db_handler import InMemoryVectorStore
from rag_embedder import (
    ChunkerConfig,
    EmbeddingPipeline,
    MockEmbedder,
    MockSparseEmbedder,
    SimpleTokenizer,
    create_chunker,
)
from rag_retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ReciprocalRankFusion,
    RetrievalConfig,
)

DOC_TEXT = """Solar array maintenance runbook.

Clean panels monthly in dusty climates; quarterly elsewhere.
The inverter fault code F17 indicates a ground-fault on string B.

Battery room: keep ambient temperature below 25 degrees Celsius.
Ventilation fans must run whenever charging current exceeds 40 amperes.

Grid export is capped at 250 kilowatts by the interconnection agreement.
"""

QUERY_TEXT = "what does inverter fault code F17 mean"


async def main() -> None:
    dim = 256
    embedder = MockEmbedder(dim=dim)
    document = Document(source_uri="solar_runbook.md", text=DOC_TEXT)

    # Chunk + embed + index into the in-memory store.
    chunker = create_chunker(
        ChunkerConfig(strategy="recursive", chunk_size=48, overlap=8, min_chunk_size=16),
        SimpleTokenizer(),
    )
    store = InMemoryVectorStore(vector_size=dim)
    pipeline = EmbeddingPipeline(
        chunker=chunker,
        embedder=embedder,
        sparse_embedder=MockSparseEmbedder(),
        store=store,
    )
    outcome = await pipeline.process(document)
    print(
        f"Indexed {outcome.chunks} chunks (dim={outcome.dims}, model={outcome.model}) "
        f"in {outcome.took_ms:.1f} ms"
    )

    query = Query(text=QUERY_TEXT, top_k=3)
    config = RetrievalConfig(strategies=["dense", "bm25"], top_k=3, candidate_k=10)
    chunks = await chunker.chunk(document)

    # Dense retrieval: embed the query, cosine-search the store.
    dense = DenseRetriever(store, embedder, config)
    dense_result = await dense.retrieve(query)
    print(f"\nDense top {len(dense_result.hits)} hits for {QUERY_TEXT!r}:")
    for hit in dense_result.hits:
        preview = " ".join((hit.text or "").split())[:55]
        print(f"  #{hit.rank} score={hit.score:.3f}  {preview!r}")

    # Hybrid: dense + BM25 fused with reciprocal rank fusion.
    corpus = [(chunk.id, document.id, chunk.text) for chunk in chunks]
    hybrid = HybridRetriever(
        retrievers=[dense, BM25Retriever(lambda: corpus, config)],
        fusion=ReciprocalRankFusion(),
        config=config,
    )
    hybrid_result = await hybrid.retrieve(query)
    print(f"\nHybrid (dense+BM25, RRF) top {len(hybrid_result.hits)} hits:")
    for hit in hybrid_result.hits:
        preview = " ".join((hit.text or "").split())[:55]
        strategies = ",".join(f"{k}={v:.2f}" for k, v in hit.strategy_scores.items())
        print(f"  #{hit.rank} score={hit.score:.3f} [{strategies}]  {preview!r}")


if __name__ == "__main__":
    asyncio.run(main())
