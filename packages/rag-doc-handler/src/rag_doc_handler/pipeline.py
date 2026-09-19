"""High-level ingestion pipeline: load -> detect -> parse -> dedup."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from rag_core.documents import Document
from rag_core.errors import UnsupportedFormatError

from rag_doc_handler.detection import detect_extension, guess_mime, sniff_mime
from rag_doc_handler.loader import SourceLoader
from rag_doc_handler.parsers.base import BaseParser, ParserRegistry, install_hint_for

__all__ = ["DedupIndex", "IngestionPipeline", "IngestionPipelineConfig"]


@dataclass(frozen=True)
class IngestionPipelineConfig:
    """Tunable knobs for :class:`IngestionPipeline`."""

    auto_detect: bool = True


class DedupIndex:
    """In-process, single-event-loop dedup store keyed by document content hash.

    Safe for concurrent use within one asyncio loop (no awaits in :meth:`add`/:meth:`has`).
    For multi-process or durable dedup, back it with a :class:`KeyValueStore`.
    """

    __slots__ = ("_hashes",)

    def __init__(self) -> None:
        self._hashes: set[str] = set()

    def add(self, content_hash: str) -> None:
        self._hashes.add(content_hash)

    def has(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    def __contains__(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    def __len__(self) -> int:
        return len(self._hashes)

    def clear(self) -> None:
        self._hashes.clear()


class IngestionPipeline:
    """Orchestrates load -> type detection -> parse -> dedup.

    Parameters
    ----------
    registry:
        Parser registry used to select a parser for the detected type.
    loader:
        Source loader used when raw ``data`` is not supplied to :meth:`ingest`.
    config:
        Optional :class:`IngestionPipelineConfig`.
    """

    def __init__(
        self,
        registry: ParserRegistry,
        loader: SourceLoader,
        config: IngestionPipelineConfig | None = None,
    ) -> None:
        self.registry = registry
        self.loader = loader
        self.config = config or IngestionPipelineConfig()

    async def ingest(
        self,
        source_uri: str,
        data: bytes | None = None,
        *,
        dedup: DedupIndex | None = None,
    ) -> tuple[Document, bool]:
        """Ingest a source.

        Returns ``(document, is_new)``. When *dedup* is provided and a document with
        the same content hash was already ingested, the (identical) document is
        returned again with ``is_new=False`` and the index is left untouched.
        """
        mime: str | None = None
        if data is None:
            loaded = await self.loader.load(source_uri)
            data = loaded.data
            source_uri = loaded.source_uri
            mime = loaded.mime_type
        else:
            mime = sniff_mime(data) or guess_mime(source_uri, data)

        parser = self._select_parser(source_uri, mime)
        document = await parser.parse(source_uri, data)

        is_new = True
        if dedup is not None:
            if dedup.has(document.content_hash):
                is_new = False
            else:
                dedup.add(document.content_hash)
        return document, is_new

    async def ingest_file(
        self, path: str | pathlib.Path, *, dedup: DedupIndex | None = None
    ) -> tuple[Document, bool]:
        """Convenience: ingest a local file via the configured loader."""
        return await self.ingest(str(path), None, dedup=dedup)

    def _select_parser(self, source_uri: str, mime: str | None) -> BaseParser:
        ext = detect_extension(source_uri)
        candidate_keys: list[str] = [ext] if ext else []
        if mime:
            candidate_keys.append(mime)
        for key in candidate_keys:
            parser = self.registry.get_or_none(key)
            if parser is not None:
                return parser
        # No parser found: raise with an install hint for known optional formats.
        target = ext or mime or ""
        hint = install_hint_for(target)
        details: dict[str, str] = {"extension": target}
        if hint is not None:
            details["install_hint"] = hint
            message = f"No parser registered for {target!r}; {hint}"
        else:
            message = f"No parser registered for {target!r}"
        raise UnsupportedFormatError(message, details=details)
