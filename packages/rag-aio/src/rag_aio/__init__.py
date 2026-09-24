"""rag-aio: RAG facade, CLI, and service composition for the rag-aio platform."""

from __future__ import annotations

from rag_aio.config import RAGConfig
from rag_aio.facade import RAG

__version__ = "0.1.0"

__all__ = ["RAG", "RAGConfig", "__version__"]
