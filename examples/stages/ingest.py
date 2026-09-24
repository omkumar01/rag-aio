"""Stage 1 — Ingestion: load, parse, and chunk a document.

Demonstrates rag-doc-handler's :class:`IngestionPipeline` (file loading, MIME
detection, Markdown parsing, dedup) followed by rag-embedder chunking with the
recursive chunker and a dependency-light tokenizer.

Fully offline; no network and no model downloads.

Run: uv run python examples/stages/ingest.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from rag_doc_handler import default_pipeline
from rag_embedder import ChunkerConfig, SimpleTokenizer, create_chunker

SAMPLE = """# rag-aio Field Guide

The hedgehog at the center of this platform never sleeps.
It patrols the vector index nightly, pruning stale points.

## Feeding schedule

Dense vectors are fed every ninety days with fresh model weights.
Sparse vectors prefer keyword hay, harvested quarterly.

## Escalations

If retrieval quality drops, page the relevance engineer on duty.
"""


async def main() -> None:
    # 1. Parse: IngestionPipeline = load -> detect type -> parse -> dedup.
    ingestion = default_pipeline()
    with tempfile.TemporaryDirectory(prefix="rag-aio-ingest-") as tmp:
        doc_path = Path(tmp) / "field_guide.md"
        doc_path.write_text(SAMPLE, encoding="utf-8")

        document, is_new = await ingestion.ingest_file(doc_path)
        print(f"Parsed {document.source_uri}")
        print(f"  parser: {document.parser_name} | new: {is_new}")
        print(f"  text: {len(document.text)} chars, {len(document.pages)} page(s)")

        # 2. Chunk: recursive strategy, token-aware, with overlap.
        chunker = create_chunker(
            ChunkerConfig(strategy="recursive", chunk_size=48, overlap=8, min_chunk_size=16),
            SimpleTokenizer(),
        )
        chunks = await chunker.chunk(document)

    token_counts = [chunk.token_count or 0 for chunk in chunks]
    print(
        f"\nChunked into {len(chunks)} chunks "
        f"(avg {sum(token_counts) / max(len(token_counts), 1):.1f} tokens)"
    )
    for chunk in chunks:
        preview = " ".join(chunk.text.split())[:60]
        print(f"  [{chunk.index:2d}] tokens={chunk.token_count:3d}  {preview!r}")


if __name__ == "__main__":
    asyncio.run(main())
