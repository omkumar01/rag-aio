"""Tests for RagManager: per-user composite stores, ingest + ask in mock mode."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from web_example.services.rag_factory import RagManager
from web_example.settings import AppSettings

_DOC_TEXT = (
    "The zebra fox naps in the sunny loft every afternoon, guarded by a very sleepy caretaker."
)
_QUESTION = "Where does the zebra fox nap?"


async def test_get_returns_cached_per_user_instances(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    rag_a1 = await manager.get(session, "user1")
    rag_a2 = await manager.get(session, "user1")
    rag_b = await manager.get(session, "user2")
    assert rag_a1 is rag_a2
    assert rag_a1 is not rag_b


async def test_per_user_instances_share_one_base_store(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    rag_a = await manager.get(session, "user1")
    rag_b = await manager.get(session, "user2")
    store_a = rag_a.services.vector_store
    store_b = rag_b.services.vector_store
    assert store_a is not None and store_b is not None
    assert store_a is not store_b
    assert store_a.base is store_b.base
    assert store_a.primary_ns == "user_user1"
    assert store_a.search_ns == ("user_user1", "global")


async def test_ingest_and_ask_end_to_end_in_mock_mode(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    document = await manager.ingest_bytes(session, "user1", "notes.md", _DOC_TEXT.encode("utf-8"))
    assert document.id

    rag_a = await manager.get(session, "user1")
    result = await rag_a.ask(_QUESTION)
    assert _QUESTION.split()[3] in result.answer  # "zebra" echoed back from context
    assert "zebra fox" in result.answer

    base = rag_a.services.vector_store
    assert base is not None
    assert base.base is not None
    assert base.base.count("user_user1") > 0
    assert base.base.count("user_user2") == 0


async def test_user_b_cannot_retrieve_user_a_document(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    await manager.ingest_bytes(session, "user1", "notes.txt", _DOC_TEXT.encode("utf-8"))
    rag_b = await manager.get(session, "user2")
    result = await rag_b.ask(_QUESTION)
    assert "zebra fox naps" not in result.answer


async def test_get_global_instance_scopes_to_global_namespace(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    rag = await manager.get_global(session)
    store = rag.services.vector_store
    assert store is not None
    assert store.primary_ns == "global"
    assert store.search_ns == ("global",)
    assert await manager.get_global(session) is rag


async def test_ingest_bytes_into_global_pool(session: AsyncSession, settings: AppSettings) -> None:
    manager = RagManager(settings)
    document = await manager.ingest_bytes(
        session, "user1", "shared.md", _DOC_TEXT.encode("utf-8"), global_pool=True
    )
    rag = await manager.get_global(session)
    store = rag.services.vector_store
    assert store is not None
    assert store.base is not None
    assert store.base.count("global") > 0
    assert document.id


async def test_invalidate_drops_cached_instances(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    rag = await manager.get(session, "user1")
    await manager.invalidate("user1")
    assert await manager.get(session, "user1") is not rag
    await manager.invalidate()
    assert await manager.get(session, "user1") is not rag


async def test_ingest_bytes_registers_chunk_counts(
    session: AsyncSession, settings: AppSettings
) -> None:
    manager = RagManager(settings)
    before = await manager.count_chunks(session, "user1")
    await manager.ingest_bytes(session, "user1", "notes.txt", _DOC_TEXT.encode("utf-8"))
    after = await manager.count_chunks(session, "user1")
    assert after > before
