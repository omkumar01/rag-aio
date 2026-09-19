"""Lightweight type/encoding detection without importing heavy parsers."""

from __future__ import annotations

import mimetypes

__all__ = ["detect_extension", "sniff_mime"]


def detect_extension(source_uri: str) -> str:
    """Return the lowercased extension (including the leading dot) of *source_uri*.

    Handles plain paths, ``file://`` URIs and URLs (query/fragment stripped) and
    Windows-style paths. Returns ``""`` when no extension is present (``"txt"``,
    ``"/path/.bashrc"``, leading-dot files).
    """
    path = source_uri
    # Drop a scheme so "file:///a/b.txt" and "http://h/x.txt" reduce to the path part.
    if "://" in path:
        path = path.split("://", 1)[1]
    # Strip query and fragment.
    for sep in ("?", "#"):
        idx = path.find(sep)
        if idx != -1:
            path = path[:idx]
    # Take the final path segment (handles both / and \\ separators).
    slash = max(path.rfind("/"), path.rfind("\\"))
    if slash != -1:
        path = path[slash + 1 :]
    dot = path.rfind(".")
    if dot <= 0:
        return ""
    return path[dot:].lower()


def sniff_mime(data: bytes) -> str | None:
    """Cheap content-sniffing for the magic-byte / prefix cases we care about.

    Returns ``None`` when the type cannot be cheaply determined; callers should then
    fall back to extension-based guessing via :func:`guess_mime` / ``mimetypes``.
    """
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"PK\x03\x04"):
        # Office Open XML and other zip-based archives: rely on the extension for the
        # specific subtype, but report a concrete zip media type here.
        return "application/zip"
    sample = data[:256].lstrip().lower()
    if sample.startswith(b"<html") or sample.startswith(b"<!doctype html"):
        return "text/html"
    if sample[:1] == b"{":
        return "application/json"
    return None


def guess_mime(source_uri: str, data: bytes | None = None) -> str | None:
    """Best-effort MIME resolution: sniff first, then extension-based fallback."""
    if data is not None:
        guessed = sniff_mime(data)
        if guessed is not None:
            return guessed
    # Strip query/fragment so mimetypes sees the real extension.
    path = source_uri
    for sep in ("?", "#"):
        idx = path.find(sep)
        if idx != -1:
            path = path[:idx]
    mime, _ = mimetypes.guess_type(path)
    return mime
