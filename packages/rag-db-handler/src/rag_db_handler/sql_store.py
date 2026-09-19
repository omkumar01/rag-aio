"""SQL-backed stores (SQLAlchemy 2.x async, sqlite default).

Provides :class:`SQLStore` (shared engine/table base + idempotent ``init_db``),
:class:`SQLDocumentStore` (the :class:`rag_core.protocols.DocumentStore` adapter)
and :class:`SQLKeyValueStore` (the :class:`rag_core.protocols.KeyValueStore`
adapter). Documents are stored with a full-JSON ``data`` column (the source of
truth for round-tripping) plus denormalized query columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.documents import Document
from rag_core.protocols import DocumentStore, KeyValueStore
from sqlalchemy import LargeBinary, Text, delete, func, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .serialization import document_to_row_fields, row_to_document

if TYPE_CHECKING:
    from .config import SQLStoreConfig


class Base(DeclarativeBase):
    """Declarative base for all SQL store tables."""


class DocumentRow(Base):
    """A stored document plus full-JSON ``data`` for lossless round-tripping."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    content_hash: Mapped[str] = mapped_column()
    source_uri: Mapped[str] = mapped_column()
    tenant: Mapped[str] = mapped_column(default="")
    namespace: Mapped[str] = mapped_column(default="")
    title: Mapped[str] = mapped_column(default="")
    mime_type: Mapped[str] = mapped_column(default="")
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column()
    updated_at: Mapped[str | None] = mapped_column()
    data: Mapped[str] = mapped_column(Text)


class KVRow(Base):
    """A namespaced key/value entry.

    The primary key is the composite ``(namespace, key)`` so that identical keys
    in different namespaces never collide (the literal column list in the ADR is
    ``key str pk`` / ``namespace str``; we widen the PK to honor namespacing).
    """

    __tablename__ = "kv"

    namespace: Mapped[str] = mapped_column(primary_key=True, default="default")
    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[bytes] = mapped_column(LargeBinary)


class SQLStore:
    """Shared SQLAlchemy async engine, session factory, and table management."""

    def __init__(self, config: SQLStoreConfig) -> None:
        self._config = config
        self._engine = create_async_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            echo=config.echo,
        )
        self._session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        self._initialized = False

    @property
    def config(self) -> SQLStoreConfig:
        return self._config

    async def init_db(self) -> None:
        """Create all tables idempotently (cached after first success)."""
        if self._initialized:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.init_db()

    async def close(self) -> None:
        await self._engine.dispose()


class SQLDocumentStore(SQLStore, DocumentStore):
    """DocumentStore backed by the ``documents`` table."""

    async def put(self, document: Document) -> None:
        await self._ensure_init()
        fields = document_to_row_fields(document)
        stmt = sqlite_insert(DocumentRow).values(**fields)
        update = {k: v for k, v in fields.items() if k != "id"}
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update)
        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()

    async def get(self, document_id: str) -> Document | None:
        await self._ensure_init()
        async with self._session() as session:
            row = (
                await session.execute(select(DocumentRow).where(DocumentRow.id == document_id))
            ).scalar_one_or_none()
        if row is None:
            return None
        return row_to_document({"data": row.data})

    async def delete(self, document_id: str) -> None:
        await self._ensure_init()
        async with self._session() as session:
            await session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))
            await session.commit()

    async def find_by_hash(self, content_hash: str) -> Document | None:
        await self._ensure_init()
        async with self._session() as session:
            row = (
                await session.execute(
                    select(DocumentRow).where(DocumentRow.content_hash == content_hash)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return row_to_document({"data": row.data})

    async def count(self) -> int:
        await self._ensure_init()
        async with self._session() as session:
            result = await session.execute(select(func.count()).select_from(DocumentRow))
            return int(result.scalar_one())

    async def health(self) -> bool:
        try:
            async with self._session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


class SQLKeyValueStore(SQLStore, KeyValueStore):
    """KeyValueStore backed by the ``kv`` table, scoped to a fixed namespace."""

    def __init__(self, config: SQLStoreConfig, namespace: str = "default") -> None:
        super().__init__(config)
        self._namespace = namespace

    async def get(self, key: str) -> bytes | None:
        await self._ensure_init()
        async with self._session() as session:
            row = (
                await session.execute(
                    select(KVRow).where(KVRow.key == key, KVRow.namespace == self._namespace)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return bytes(row.value)

    async def set(self, key: str, value: bytes) -> None:
        await self._ensure_init()
        stmt = sqlite_insert(KVRow).values(namespace=self._namespace, key=key, value=value)
        stmt = stmt.on_conflict_do_update(
            index_elements=["namespace", "key"], set_={"value": value}
        )
        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete(self, key: str) -> None:
        await self._ensure_init()
        async with self._session() as session:
            await session.execute(
                delete(KVRow).where(KVRow.key == key, KVRow.namespace == self._namespace)
            )
            await session.commit()

    async def health(self) -> bool:
        try:
            async with self._session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


__all__ = ["Base", "SQLDocumentStore", "SQLKeyValueStore", "SQLStore"]
