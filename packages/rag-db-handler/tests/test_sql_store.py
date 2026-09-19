"""Integration tests for SQL-backed DocumentStore and KeyValueStore (sqlite)."""

from __future__ import annotations

import pytest
from rag_core.documents import Document, DocumentMetadata
from rag_core.protocols import DocumentStore, KeyValueStore
from rag_db_handler import SQLStoreConfig, create_kv_store, create_sql_store
from rag_db_handler.sql_store import SQLDocumentStore, SQLKeyValueStore

pytestmark = pytest.mark.integration


def _doc(source: str = "mem://doc", text: str = "hello world") -> Document:
    return Document(
        source_uri=source, text=text, metadata=DocumentMetadata(title="T", mime_type="text/plain")
    )


@pytest.fixture()
async def doc_store(tmp_path) -> SQLDocumentStore:  # type: ignore[no-untyped-def]
    cfg = SQLStoreConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'doc.db'}")
    store = create_sql_store(cfg)
    await store.init_db()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture()
async def kv_store(tmp_path) -> SQLKeyValueStore:  # type: ignore[no-untyped-def]
    cfg = SQLStoreConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'kv.db'}")
    store = SQLKeyValueStore(cfg, namespace="ns")
    await store.init_db()
    try:
        yield store
    finally:
        await store.close()


async def test_protocol_conformance(
    doc_store: SQLDocumentStore, kv_store: SQLKeyValueStore
) -> None:
    assert isinstance(doc_store, DocumentStore)
    assert isinstance(kv_store, KeyValueStore)


async def test_put_get_roundtrip(doc_store: SQLDocumentStore) -> None:
    doc = _doc(text="the quick brown fox")
    await doc_store.put(doc)
    fetched = await doc_store.get(doc.id)
    assert fetched is not None
    assert fetched.id == doc.id
    assert fetched.source_uri == doc.source_uri
    assert fetched.text == "the quick brown fox"
    assert fetched.metadata.title == "T"
    assert fetched.metadata.mime_type == "text/plain"
    # content_hash is derived and must round-trip
    assert fetched.content_hash == doc.content_hash
    assert fetched.created_at == doc.created_at


async def test_put_updates_overwrites(doc_store: SQLDocumentStore) -> None:
    doc = _doc(text="v1")
    await doc_store.put(doc)
    doc2 = Document(id=doc.id, source_uri=doc.source_uri, text="v2", metadata=DocumentMetadata())
    await doc_store.put(doc2)
    fetched = await doc_store.get(doc.id)
    assert fetched is not None
    assert fetched.text == "v2"


async def test_find_by_hash(doc_store: SQLDocumentStore) -> None:
    doc = _doc(text="some content")
    await doc_store.put(doc)
    found = await doc_store.find_by_hash(doc.content_hash)
    assert found is not None
    assert found.id == doc.id
    missing = await doc_store.find_by_hash("0" * 64)
    assert missing is None


async def test_delete(doc_store: SQLDocumentStore) -> None:
    doc = _doc()
    await doc_store.put(doc)
    assert await doc_store.get(doc.id) is not None
    await doc_store.delete(doc.id)
    assert await doc_store.get(doc.id) is None


async def test_health_and_count(doc_store: SQLDocumentStore) -> None:
    assert await doc_store.health() is True
    assert await doc_store.count() == 0
    await doc_store.put(_doc())
    await doc_store.put(_doc(text="second"))
    assert await doc_store.count() == 2


async def test_kv_roundtrip(kv_store: SQLKeyValueStore) -> None:
    assert await kv_store.get("missing") is None
    await kv_store.set("k1", b"value-bytes")
    assert await kv_store.get("k1") == b"value-bytes"
    await kv_store.set("k1", b"updated")
    assert await kv_store.get("k1") == b"updated"
    await kv_store.delete("k1")
    assert await kv_store.get("k1") is None


async def test_kv_namespace_isolation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = SQLStoreConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'ns.db'}")
    s1 = SQLKeyValueStore(cfg, namespace="a")
    s2 = SQLKeyValueStore(cfg, namespace="b")
    await s1.init_db()
    await s2.init_db()
    try:
        await s1.set("k", b"from-a")
        await s2.set("k", b"from-b")
        assert await s1.get("k") == b"from-a"
        assert await s2.get("k") == b"from-b"
    finally:
        await s1.close()
        await s2.close()


async def test_kv_created_via_factory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from rag_db_handler import KeyValueStoreConfig

    store = create_kv_store(
        KeyValueStoreConfig(backend="sqlite", path=str(tmp_path / "f.kv"), namespace="f")
    )
    await store.init_db()
    try:
        await store.set("k", b"v")
        assert await store.get("k") == b"v"
    finally:
        await store.close()
