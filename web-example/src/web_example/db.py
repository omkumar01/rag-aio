"""Async SQLAlchemy data layer for the web-example app (users, chats, docs, jobs).

This is the *web* database (accounts, settings, chat history, the document
registry, bulk-upload job tracking). RAG content itself lives in the vector
store and document store configured through :mod:`web_example.services`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_core.ids import new_id
from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "Base",
    "Conversation",
    "Document",
    "Message",
    "SettingRow",
    "UploadJob",
    "User",
    "init_db",
    "sessionmaker_from",
]


def utcnow() -> datetime:
    """Naive-UTC now (sqlite-friendly)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SettingRow(Base):
    """Global defaults and per-user overrides, keyed (e.g. ``global``, ``user:<id>``)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text, default="")
    attachments: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Document(Base):
    """Registry row for every ingested document (metadata only; text lives in RAG stores)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # rag Document.id
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    visibility: Mapped[str] = mapped_column(String(8), default="user")  # user | global
    name: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str] = mapped_column(Text, default="")
    ext: Mapped[str] = mapped_column(String(16), default="")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    num_chunks: Mapped[int] = mapped_column(BigInteger, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class UploadJob(Base):
    """Bulk-injection job tracking (mirrors rag-mass-inject job state)."""

    __tablename__ = "upload_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    target: Mapped[str] = mapped_column(String(8), default="user")  # user | global
    kind: Mapped[str] = mapped_column(String(16), default="files")  # files | directory | url
    status: Mapped[str] = mapped_column(String(16), default="queued")
    total: Mapped[int] = mapped_column(BigInteger, default=0)
    processed: Mapped[int] = mapped_column(BigInteger, default=0)
    failed: Mapped[int] = mapped_column(BigInteger, default=0)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


def sessionmaker_from(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(app: Any) -> None:
    """Create the engine + schema and stash both on ``app.state``."""
    data_dir = Path(app.state.settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    engine = create_async_engine(app.state.settings.db_url)
    app.state.engine = engine
    app.state.sessionmaker = sessionmaker_from(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
