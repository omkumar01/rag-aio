"""FastAPI application factory for the rag-aio facade.

Importing this module builds the mock (offline) application eagerly so that
``uvicorn rag_aio.app:app`` works out of the box.  Heavy backends are only
imported inside :func:`rag_aio.facade.build_services` /
:func:`rag_aio.facade._build_mock_services`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_aio.config import RAGConfig
from rag_aio.facade import build_services

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["app", "get_app"]


def get_app(config_path: str | None = None) -> FastAPI:
    """Create a FastAPI app from a TOML config file (or mock defaults)."""
    from rag_orchestrator.app.app import create_app

    config = RAGConfig.from_file(config_path) if config_path else RAGConfig.mock()
    services = build_services(config)
    return create_app(services, config.pipeline)


# Module-level instance for ``uvicorn rag_aio.app:app``.
app: FastAPI = get_app()
