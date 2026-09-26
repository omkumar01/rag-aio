"""Unit tests for friendly chat error messages (dimension mismatch etc.)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rag_core import RetrievalError
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import User
from web_example.services.chat_service import stream_chat


class _FailingRag:
    """RAG facade double whose ask fails exactly like a dim-mismatched store."""

    services = SimpleNamespace(generator=None)

    async def ask(self, query: str, *, stream: bool = False) -> Any:
        try:
            raise ValueError(
                "shapes (505,384) and (1024,) not aligned: 384 (dim 1) != 1024 (dim 0)"
            )
        except ValueError as exc:
            raise RetrievalError("all 1 retrieval pairs failed") from exc


class _FailingManager:
    async def get(self, session: AsyncSession, user_id: str) -> _FailingRag:
        return _FailingRag()


async def test_dimension_mismatch_yields_friendly_error(session: AsyncSession) -> None:
    user = User(username="erruser", password_hash="x")
    session.add(user)
    await session.commit()

    events = [
        event async for event in stream_chat(session, _FailingManager(), user, None, "hello", [])
    ]

    kinds = [event["type"] for event in events]
    assert kinds[0] == "meta"
    assert "error" in kinds
    assert "done" not in kinds
    error = next(event for event in events if event["type"] == "error")
    assert "dimension mismatch" in error["message"]
    assert "Re-ingest" in error["message"]


async def test_generic_failure_yields_clipped_error(session: AsyncSession) -> None:
    class _BrokenRag:
        services = SimpleNamespace(generator=None)

        async def ask(self, query: str, *, stream: bool = False) -> Any:
            raise RuntimeError("boom")

    class _BrokenManager:
        async def get(self, session: AsyncSession, user_id: str) -> _BrokenRag:
            return _BrokenRag()

    user = User(username="erruser2", password_hash="x")
    session.add(user)
    await session.commit()

    events = [
        event async for event in stream_chat(session, _BrokenManager(), user, None, "hello", [])
    ]
    error = next(event for event in events if event["type"] == "error")
    assert error["message"] == "boom"
