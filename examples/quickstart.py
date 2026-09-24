"""Quickstart: the README flow, fully offline.

Loads a RAG instance from a TOML config file (the ``RAG.from_config("config.toml")``
form), ingests a small documents directory, asks a question, and prints the answer,
citations, and per-stage timings.

Uses the mock profile declared in ``quickstart_config.toml`` so it runs with no
network, no downloads, and no LM Studio. The "answer" comes from the stub
generator echoing the assembled context — it demonstrates the pipeline plumbing,
not answer quality.

Run: uv run python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from rag_aio import RAG
from rag_orchestrator import AskResult

CONFIG_PATH = Path(__file__).resolve().parent / "quickstart_config.toml"

DOC_1 = """# Alpha Project Handbook

The Alpha project ships a command-line utility called alpha-ctl.
The activation key for the Alpha staging cluster is ALP-2026-KX.
Rotation happens every ninety days; the on-call engineer owns rotation.
"""

DOC_2 = """# Bravo Project Handbook

Bravo is the data-pipeline service feeding the Alpha dashboards.
Its staging URL is https://bravo.example.internal and it deploys weekly.
Escalations page the data-platform rota, then the platform lead.
"""


async def main() -> None:
    rag = RAG.from_config(str(CONFIG_PATH))
    async with rag:
        with tempfile.TemporaryDirectory(prefix="rag-aio-quickstart-") as tmp:
            docs_dir = Path(tmp) / "documents"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(DOC_1, encoding="utf-8")
            (docs_dir / "bravo.md").write_text(DOC_2, encoding="utf-8")

            ingested = await rag.ingest_directory(docs_dir)
            print(f"Ingested {len(ingested)} document(s):")
            for doc in ingested:
                print(f"  - {doc.source_uri} ({len(doc.text)} chars, id={doc.id})")

            result = await rag.ask("What is the activation key for the Alpha staging cluster?")
            assert isinstance(result, AskResult), "expected a non-streaming AskResult"

            print("\nAnswer:")
            print(result.answer)
            print("\nCitations:")
            for citation in result.citations:
                print(
                    f"  [{citation.citation_id}] {citation.source_uri or citation.document_id}"
                    f" pages={citation.page_numbers}"
                )
            print("\nStage timings (ms):")
            for stage, ms in result.timings_ms.items():
                print(f"  {stage:12s} {ms:8.2f}")


if __name__ == "__main__":
    asyncio.run(main())
