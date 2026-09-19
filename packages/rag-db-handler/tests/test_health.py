"""Tests for the check_health utility."""

from __future__ import annotations

from typing import Any

import pytest
from rag_db_handler import InMemoryKeyValueStore, InMemoryVectorStore, check_health


@pytest.mark.unit
async def test_check_health_all_healthy() -> None:
    vec = InMemoryVectorStore(vector_size=4)
    kv = InMemoryKeyValueStore(namespace="n")
    result = await check_health({"vector": vec, "kv": kv})
    assert result == {"vector": True, "kv": True}


@pytest.mark.unit
async def test_check_health_marks_unhealthy() -> None:
    class BadStore:
        async def health(self) -> bool:
            raise RuntimeError("boom")

    result = await check_health({"bad": BadStore()})
    assert result == {"bad": False}


@pytest.mark.unit
async def test_check_health_store_without_health_is_true() -> None:
    class StoreWithoutHealth:
        pass

    result = await check_health({"plain": StoreWithoutHealth()})
    assert result == {"plain": True}


@pytest.mark.unit
async def test_check_health_typed_input() -> None:
    stores: dict[str, Any] = {"vec": InMemoryVectorStore(vector_size=4)}
    result = await check_health(stores)
    assert result == {"vec": True}
