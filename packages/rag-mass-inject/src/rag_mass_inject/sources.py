"""Source discovery: directories, file lists, and XML sitemaps.

These helpers turn a *source* (local directory, file-list text, or sitemap URL)
into a flat list of concrete file paths / URLs that the ingestion pipeline can
process one-by-one.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as _ET

import httpx

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "discover_directory",
    "discover_file_list",
    "discover_sitemap",
    "is_supported",
]

#: File extensions understood by the doc-handler parsers.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".txt", ".md", ".html", ".htm", ".docx", ".xlsx", ".pptx", ".eml"}
)


def discover_directory(path: str | Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of regular files under *path*.

    When *recursive* is ``True`` sub-directories are traversed via
    :meth:`Path.rglob`; otherwise only the top level is scanned with
    :meth:`Path.iterdir`.

    Raises :class:`NotADirectoryError` when *path* is not a directory.
    """
    base = Path(path)
    if not base.is_dir():
        msg = f"not a directory: {path}"
        raise NotADirectoryError(msg)
    entries = base.rglob("*") if recursive else base.iterdir()
    files = [p for p in entries if p.is_file()]
    return sorted(files)


def discover_file_list(path: str | Path) -> list[str]:
    """Read a text file containing one source path/URL per line.

    Blank lines and lines starting with ``#`` are skipped.  Each remaining
    line is stripped and returned verbatim.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    result: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        result.append(stripped)
    return result


async def discover_sitemap(url: str, client: httpx.AsyncClient | None = None) -> list[str]:
    """Fetch an XML sitemap and return the URLs listed in ``<loc>`` elements.

    Uses :mod:`httpx` with a short timeout.  On any HTTP error an empty list
    is returned (callers should treat empty as "no URLs discovered").

    When *client* is supplied (e.g. one configured with
    :class:`httpx.MockTransport` in tests) it is used as-is and **not**
    closed by this function; the caller owns its lifecycle.
    """
    text = await _fetch_sitemap_text(url, client)
    return _parse_locs(text)


async def _fetch_sitemap_text(url: str, external_client: httpx.AsyncClient | None) -> str:
    """Return the raw text of the sitemap at *url*, or ``""`` on HTTP errors."""
    if external_client is not None:
        try:
            resp = await external_client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError:
            return ""
        return resp.text

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError:
            return ""
        return resp.text


def _parse_locs(xml: str) -> list[str]:
    """Extract ``<loc>`` text from a sitemap XML string (namespace-agnostic)."""
    locs: list[str] = []
    try:
        root = _ET.fromstring(xml.encode("utf-8"))
    except _ET.ParseError:
        return locs
    for el in root.iter():
        if el.tag.endswith("loc") and el.text:
            locs.append(el.text.strip())
    return locs


def is_supported(path: str | Path) -> bool:
    """Return ``True`` when *path* has a recognized document extension."""
    ext = Path(path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS
