"""SQLite-backed cache with per-entry TTL (WAL mode)."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS cache ("
    "  key TEXT PRIMARY KEY,"
    "  value BLOB NOT NULL,"
    "  expires_at REAL"
    ")"
)
_BUSY_TIMEOUT = 5.0  # seconds; backs off concurrent writers instead of raising


class SQLiteCache:
    """Persistent cache backed by a SQLite database.

    Connection strategy: a fresh :func:`sqlite3.connect` is opened for *each*
    operation, used by exactly one worker thread for the duration of that call,
    and then closed. Those connections are opened and closed inside
    :func:`asyncio.to_thread`. Because no connection is ever shared across
    threads, the SQLite cross-thread constraint is satisfied without
    ``check_same_thread=False`` and without a process-wide lock: a per-operation
    connection is the simplest and safest option for a single-writer cache.

    The database is opened in WAL journal mode (set once at construction) with a
    ``busy_timeout`` so readers never block writers and concurrent writers back
    off instead of erroring. Expiry uses wall-clock ``time.time()`` so TTLs
    survive restarts; ``expires_at IS NULL`` means "no expiry".
    """

    def __init__(self, path: str | Path, default_ttl: float | None = None) -> None:
        self._path = str(path)
        self._default_ttl = default_ttl
        self._init_schema()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT)

    def _execute_sync(self, query: str, params: tuple[Any, ...] = ()) -> None:
        conn = self._connect()
        try:
            conn.execute(query, params)
            conn.commit()
        finally:
            conn.close()

    def _query_sync(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        conn = self._connect()
        try:
            cursor = conn.execute(query, params)
            row: tuple[Any, ...] | None = cursor.fetchone()
            return row
        finally:
            conn.close()

    async def get(self, key: str) -> bytes | None:
        row = await asyncio.to_thread(
            self._query_sync, "SELECT value, expires_at FROM cache WHERE key=?", (key,)
        )
        if row is None:
            return None
        value, expires_at = row
        if expires_at is not None and time.time() >= expires_at:
            await asyncio.to_thread(self._execute_sync, "DELETE FROM cache WHERE key=?", (key,))
            return None
        return bytes(value)

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        if ttl is None:
            ttl = self._default_ttl
        expires_at = time.time() + ttl if ttl is not None else None
        await asyncio.to_thread(
            self._execute_sync,
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, expires_at),
        )

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._execute_sync, "DELETE FROM cache WHERE key=?", (key,))

    async def clear(self) -> None:
        await asyncio.to_thread(self._execute_sync, "DELETE FROM cache")

    async def size(self) -> int:
        row = await asyncio.to_thread(self._query_sync, "SELECT COUNT(*) FROM cache")
        return int(row[0]) if row else 0
