"""Command-line interface for the rag-aio facade.

Provides four commands — ``ingest``, ``ask``, ``serve``, and ``config`` — that
mirror the :class:`~rag_aio.facade.RAG` API with zero configuration needed
(mock backends are used by default).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from rag_aio.config import RAGConfig
from rag_aio.facade import RAG

app = typer.Typer(help="rag-aio: RAG facade CLI")

__all__ = ["app"]


def _load_config(config_path: Path | None) -> RAGConfig:
    if config_path is not None:
        return RAGConfig.from_file(str(config_path))
    return RAGConfig.mock()


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to the file to ingest."),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to a RAG config TOML file."
    ),
) -> None:
    """Ingest a document file into the vector store."""
    cfg = _load_config(config_path)
    rag = RAG.from_config(cfg)
    doc = asyncio.run(rag.ingest(path))
    typer.echo(f"Ingested {path}: {doc.id} ({len(doc.pages)} page(s), hash={doc.content_hash[:8]})")


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #


@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask."),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to a RAG config TOML file."
    ),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response."),
) -> None:
    """Ask a question through the RAG pipeline."""
    cfg = _load_config(config_path)
    rag = RAG.from_config(cfg)
    from rag_orchestrator import AskResult  # lazy: facade validates presence first

    if stream:

        async def _stream() -> None:
            result = await rag.ask(query, stream=True)
            async for delta in result:  # type: ignore[union-attr]
                print(delta, end="", flush=True)
            print()

        asyncio.run(_stream())
        return

    result = asyncio.run(rag.ask(query))
    assert isinstance(result, AskResult), "non-streaming ask must return AskResult"
    typer.echo(result.answer)
    for citation in result.citations:
        location = citation.source_uri or citation.document_id
        typer.echo(f"[{citation.citation_id}] {location}")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@app.command("config")
def show_config(
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to a RAG config TOML file."
    ),
) -> None:
    """Print the resolved RAG configuration as JSON."""
    cfg = _load_config(config_path)
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2))


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to a RAG config TOML file."
    ),
) -> None:
    """Start the rag-aio FastAPI server."""
    import uvicorn

    from rag_aio.app import get_app

    application = get_app(str(config_path) if config_path else None)
    uvicorn.run(application, host=host, port=port)
