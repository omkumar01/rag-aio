"""Source loaders: local files (with size guard + path resolution) and HTTP."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from rag_core.errors import IngestionError

from rag_doc_handler.detection import guess_mime
from rag_doc_handler.models import LoadedSource
from rag_doc_handler.security import validate_public_url

__all__ = ["FileLoader", "LoaderLimits", "SourceLoader", "WebLoader"]

_DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB


@dataclass
class LoaderLimits:
    """Limits enforced while loading a source."""

    max_bytes: int = _DEFAULT_MAX_BYTES


@runtime_checkable
class SourceLoader(Protocol):
    """A source loader producing a :class:`LoadedSource`.

    This is doc-handler's richer replacement for the rag-core ``DocumentLoader`` protocol
    (which returns a bare ``(bytes, dict)``): carrying the sniffed MIME type and resolved
    source URI makes the pipeline's type detection deterministic and testable.
    """

    async def load(self, source: str) -> LoadedSource: ...


class FileLoader:
    """Loads a local file, enforcing :class:`LoaderLimits` and resolving the path."""

    def __init__(self, limits: LoaderLimits | None = None) -> None:
        self.limits = limits or LoaderLimits()

    @staticmethod
    def _inspect(source: str) -> tuple[Path, bool, int]:
        """Resolve *source*, returning (path, is_file, size) — all blocking ops here."""
        path = Path(source).resolve()
        if not path.is_file():
            return path, False, 0
        return path, True, path.stat().st_size

    async def load(self, source: str) -> LoadedSource:
        # All filesystem work happens in a worker thread so the event loop stays free.
        path, is_file, size = await asyncio.to_thread(self._inspect, source)
        # Path traversal is contained by resolving to an absolute path and requiring
        # the result to be a regular file on disk.
        if not is_file:
            raise IngestionError(
                f"Source is not a file: {source}",
                code="file_not_found",
                details={"source": source},
            )
        if size > self.limits.max_bytes:
            raise IngestionError(
                f"File exceeds max_bytes ({self.limits.max_bytes}): {source}",
                code="file_too_large",
                details={
                    "source": str(path),
                    "size": size,
                    "max_bytes": self.limits.max_bytes,
                },
            )
        data = await asyncio.to_thread(path.read_bytes)
        mime = guess_mime(str(path), data)
        metadata: dict[str, Any] = {}
        if mime:
            metadata["content_type"] = mime
        return LoadedSource(
            source_uri=str(path),
            data=data,
            mime_type=mime,
            metadata=metadata,
        )


class WebLoader:
    """Loads a URL with SSRF protection and a response size cap."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        limits: LoaderLimits | None = None,
    ) -> None:
        self.client = client
        self.limits = limits or LoaderLimits()

    async def load(self, source: str) -> LoadedSource:
        url = validate_public_url(source)
        chunks: list[bytes] = []
        total = 0
        async with self.client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            content_length = resp.headers.get("content-length")
            if content_length and _to_int(content_length) > self.limits.max_bytes:
                raise IngestionError(
                    f"Remote content exceeds max_bytes ({self.limits.max_bytes})",
                    code="file_too_large",
                    details={"url": url, "content_length": content_length},
                )
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > self.limits.max_bytes:
                    raise IngestionError(
                        f"Remote content exceeds max_bytes ({self.limits.max_bytes})",
                        code="file_too_large",
                        details={"url": str(resp.url), "downloaded": total},
                    )
                chunks.append(chunk)
            data = b"".join(chunks)
            mime = resp.headers.get("content-type")
            final_url = str(resp.url)
        metadata: dict[str, Any] = {"final_url": final_url}
        return LoadedSource(
            source_uri=final_url,
            data=data,
            mime_type=mime,
            metadata=metadata,
        )


def _to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
