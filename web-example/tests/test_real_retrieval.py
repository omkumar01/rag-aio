"""Integration test: real (fastembed) retrieval over per-user namespaces.

Skipped unless the optional fastembed extra is installed; the embedding model
itself is loaded from the local fastembed cache on first use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.services.rag_factory import RagManager
from web_example.services.settings_store import save_global_config
from web_example.settings import AppSettings


@pytest.mark.integration
async def test_real_embedder_retrieves_uploaded_document(
    session: AsyncSession, tmp_path: Path
) -> None:
    pytest.importorskip("fastembed")
    pytest.importorskip("qdrant_client")
    settings = AppSettings(data_dir=tmp_path / "data", session_secret="test-secret")
    settings.ensure_dirs()

    await save_global_config(
        session,
        {
            "embedder": {
                "backend": "fastembed",
                "dense_model": "BAAI/bge-small-en-v1.5",
                "dim": 384,
            },
            "storage": {
                "qdrant_path": str(tmp_path / "qdrant"),
                "db_url": f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}",
            },
        },
    )

    manager = RagManager(settings, config_provider=settings_store_provider(session))
    await manager.ingest_bytes(
        session, "user-1", "facts.txt", b"The Eiffel Tower is located in Paris, France."
    )
    rag = await manager.get(session, "user-1")

    from rag_core import Query

    hybrid = rag.services.retrievers[0]
    result = await hybrid.retrieve(Query(text="Where is the Eiffel Tower?", id="q1"))
    texts = [hit.text or "" for hit in result.hits]
    assert any("Eiffel Tower" in text for text in texts), texts[:3]


def settings_store_provider(session: AsyncSession):
    """Bound config_provider shim for this test (no app/request context)."""
    from web_example.services import settings_store

    async def _provider(s: AsyncSession, user_id: str):
        return await settings_store.config_provider(s, user_id)

    return _provider
