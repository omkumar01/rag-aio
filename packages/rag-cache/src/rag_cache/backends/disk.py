"""Filesystem-backed cache with per-entry TTL.

``DiskCache`` persists entries under ``directory`` in a sharded layout so that no
single directory grows unbounded:

    <directory>/<hash[:2]>/<sanitized-key>

The "hash" is the 64-hex digest produced by
:func:`rag_cache.keys.cache_key` (the part after the namespace colon); the
shard is its first two hex characters. The on-disk file is the *full* sanitized
key (``namespace:hash`` with ``:``/``/``/``\\`` replaced by ``-``) so two
namespaces with identical parts never collide.

Entry file format (single blob, written atomically via temp-file + rename):

    [8-byte big-endian float64 expiry][payload bytes]

A header of ``0.0`` means "no expiry". Using wall-clock ``time.time()`` for the
expiry lets TTLs survive process restarts. Expired entries are deleted lazily
on the next ``get``; ``max_bytes`` (when set) trims the oldest files by mtime
after each write.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path

# 8-byte big-endian IEEE-754 float64. 0.0 is the "no expiry" sentinel.
_EXPIRY_FORMAT = ">d"
_EXPIRY_SIZE = struct.calcsize(_EXPIRY_FORMAT)
_NO_EXPIRY = 0.0


class DiskCache:
    """Persistent filesystem cache.

    Args:
        directory: Root directory for cache files. Created if missing.
        max_bytes: Optional approximate byte budget. Enforced best-effort by
            evicting the oldest files (by modification time) after a write.
            ``None`` disables byte-based eviction.
        default_ttl: Default TTL (seconds) for entries set without an explicit
            ``ttl``. ``None`` means no expiry unless a TTL is supplied.
    """

    def __init__(
        self,
        directory: str | Path,
        max_bytes: int | None = None,
        default_ttl: float | None = None,
    ) -> None:
        self._root = Path(directory)
        self._max_bytes = max_bytes if max_bytes and max_bytes > 0 else None
        self._default_ttl = default_ttl
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Resolve a key to its on-disk path, sharded by the hash's first hex chars."""
        # key == "namespace:<hex digest>"; digest lives after the first colon.
        _, _, digest = key.partition(":")
        shard = digest[:2] if digest else key[:2]
        safe = key.replace(":", "-").replace("/", "-").replace("\\", "-")
        return self._root / shard / safe

    def _entry_files(self) -> list[Path]:
        return [p for p in self._root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]

    async def get(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        if len(data) < _EXPIRY_SIZE:
            # Corrupt/truncated entry: drop it and treat as a miss.
            path.unlink(missing_ok=True)
            return None
        (expires_at,) = struct.unpack(_EXPIRY_FORMAT, data[:_EXPIRY_SIZE])
        if expires_at != _NO_EXPIRY and time.time() >= expires_at:
            path.unlink(missing_ok=True)
            return None
        return data[_EXPIRY_SIZE:]

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        await asyncio.to_thread(self._set_sync, key, value, ttl)

    def _set_sync(self, key: str, value: bytes, ttl: float | None) -> None:
        if ttl is None:
            ttl = self._default_ttl
        expires_at = time.time() + ttl if ttl is not None else _NO_EXPIRY
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = struct.pack(_EXPIRY_FORMAT, expires_at) + value
        self._atomic_write(path, payload)
        self._enforce_max_bytes()

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        """Write ``payload`` to ``path`` atomically (temp file + rename)."""
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="ragcache_", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _enforce_max_bytes(self) -> None:
        if self._max_bytes is None:
            return
        files = sorted(self._entry_files(), key=lambda p: p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self._max_bytes:
                break
            with contextlib.suppress(OSError):
                total -= path.stat().st_size
                path.unlink()

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    async def clear(self) -> None:
        await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        if self._root.exists():
            shutil.rmtree(self._root)
        self._root.mkdir(parents=True, exist_ok=True)

    async def size(self) -> int:
        return await asyncio.to_thread(self._size_sync)

    def _size_sync(self) -> int:
        return len(self._entry_files())
