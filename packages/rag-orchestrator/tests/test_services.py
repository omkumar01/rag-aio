"""Tests for ``OrchestratorServices.aclose()`` (offline: fakes only)."""

from __future__ import annotations

import asyncio

from rag_orchestrator.services import OrchestratorServices


class AsyncAcloser:
    """Component exposing ``async def aclose``."""

    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1
        self.log.append(self.name)


class SyncCloser:
    """Component exposing a plain synchronous ``close``."""

    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.log.append(self.name)


class SyncReturningCoroutineCloser:
    """Component whose ``close`` returns a coroutine (async work, sync def)."""

    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name
        self.close_count = 0

    def close(self):
        async def _close() -> None:
            self.close_count += 1
            self.log.append(self.name)

        return _close()


class NotClosable:
    """Component without any close method — must be skipped silently."""


def _make(log: list[str]) -> OrchestratorServices:
    return OrchestratorServices(
        vector_store=AsyncAcloser(log, "vector_store"),  # type: ignore[arg-type]
        document_store=SyncCloser(log, "document_store"),  # type: ignore[arg-type]
        cache=SyncReturningCoroutineCloser(log, "cache"),  # type: ignore[arg-type]
        embedder=AsyncAcloser(log, "embedder"),  # type: ignore[arg-type]
        query_engine=NotClosable(),  # type: ignore[arg-type]
    )


def test_aclose_closes_primaries_then_other_components_in_order() -> None:
    log: list[str] = []
    services = _make(log)
    asyncio.run(services.aclose())
    assert log == ["vector_store", "document_store", "cache", "embedder"]
    assert isinstance(services.query_engine, NotClosable)  # untouched, no error


def test_aclose_is_idempotent() -> None:
    log: list[str] = []
    services = _make(log)
    asyncio.run(services.aclose())
    asyncio.run(services.aclose())
    assert log == ["vector_store", "document_store", "cache", "embedder"]
    assert isinstance(services.vector_store, AsyncAcloser)
    assert services.vector_store.close_count == 1
    assert isinstance(services.document_store, SyncCloser)
    assert services.document_store.close_count == 1


def test_aclose_on_empty_services_is_a_noop() -> None:
    services = OrchestratorServices()
    asyncio.run(services.aclose())  # must not raise
    asyncio.run(services.aclose())


def test_aclose_accepts_none_components() -> None:
    services = OrchestratorServices(
        vector_store=None,
        document_store=None,
        cache=SyncCloser([], "cache"),  # type: ignore[arg-type]
    )
    asyncio.run(services.aclose())  # must not raise on None fields
