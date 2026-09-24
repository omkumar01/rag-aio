"""Tests for source discovery: directories, file lists, sitemaps, helpers."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from rag_mass_inject.sources import (
    SUPPORTED_EXTENSIONS,
    discover_directory,
    discover_file_list,
    discover_sitemap,
    is_supported,
)

# -- discover_directory -------------------------------------------------------


def test_discover_directory_flat(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")

    result = discover_directory(str(tmp_path), recursive=False)
    names = sorted(p.name for p in result)
    assert names == ["a.txt", "b.md", "c.bin"]


def test_discover_directory_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "nested"
    sub.mkdir()
    deep = sub / "deep"
    deep.mkdir()

    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    (sub / "mid.md").write_text("mid", encoding="utf-8")
    (deep / "bottom.txt").write_text("bottom", encoding="utf-8")

    result = discover_directory(str(tmp_path), recursive=True)
    assert len(result) == 3
    # Sorted by full path, so check all parents are present
    all_paths = sorted(str(p) for p in result)
    assert any("top.txt" in p for p in all_paths)
    assert any("mid.md" in p for p in all_paths)
    assert any("bottom.txt" in p for p in all_paths)


def test_discover_directory_recursive_false_skips_subdirs(tmp_path: Path) -> None:
    sub = tmp_path / "nested"
    sub.mkdir()
    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    (sub / "mid.md").write_text("mid", encoding="utf-8")

    result = discover_directory(str(tmp_path), recursive=False)
    assert len(result) == 1
    assert result[0].name == "top.txt"


def test_discover_directory_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        discover_directory(str(tmp_path / "nonexistent"), recursive=False)


# -- discover_file_list -------------------------------------------------------


def test_discover_file_list(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "# comment line",
            "",
            "  /path/to/doc1.txt  ",
            "https://example.com/page.html",
            "",
            "# another comment",
            "doc2.md",
        ]
    )
    fpath = tmp_path / "files.txt"
    fpath.write_text(content, encoding="utf-8")

    result = discover_file_list(str(fpath))
    assert result == ["/path/to/doc1.txt", "https://example.com/page.html", "doc2.md"]


def test_discover_file_list_empty(tmp_path: Path) -> None:
    fpath = tmp_path / "empty.txt"
    fpath.write_text("# only comments\n\n", encoding="utf-8")
    assert discover_file_list(str(fpath)) == []


# -- discover_sitemap ---------------------------------------------------------

_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
  <url>
    <loc>https://example.com/page3</loc>
  </url>
</urlset>
"""


def _make_mock_client(xml_text: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=xml_text, headers={"content-type": "application/xml"})

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


async def test_discover_sitemap_parses_locs(tmp_path: Path) -> None:
    client = _make_mock_client(_SITEMAP_XML)
    try:
        urls = await discover_sitemap("https://example.com/sitemap.xml", client=client)
    finally:
        await client.aclose()
    assert urls == [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]


async def test_discover_sitemap_http_error_returns_empty(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    try:
        urls = await discover_sitemap("https://example.com/sitemap.xml", client=client)
    finally:
        await client.aclose()
    assert urls == []


async def test_discover_sitemap_no_client_uses_default(tmp_path: Path) -> None:
    """When no client is supplied, a default AsyncClient is created (no assertion on network)."""
    # This test verifies the no-client path doesn't crash on invalid URL.
    urls = await discover_sitemap("http://localhost:1/sitemap.xml")
    assert urls == []


# -- is_supported -------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("doc.pdf", True),
        ("doc.txt", True),
        ("doc.md", True),
        ("doc.html", True),
        ("doc.htm", True),
        ("doc.docx", True),
        ("doc.xlsx", True),
        ("doc.pptx", True),
        ("doc.eml", True),
        ("doc.PDF", True),  # case-insensitive
        ("doc.bin", False),
        ("doc.xyz", False),
        ("doc", False),
    ],
)
def test_is_supported(path: str, expected: bool) -> None:
    assert is_supported(path) is expected


def test_supported_extensions_nonempty() -> None:
    assert len(SUPPORTED_EXTENSIONS) >= 8
