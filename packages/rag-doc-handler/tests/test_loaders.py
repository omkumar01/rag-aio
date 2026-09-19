"""Tests for FileLoader and WebLoader."""

from __future__ import annotations

import httpx
import pytest
from rag_core.errors import IngestionError
from rag_doc_handler.loader import FileLoader, LoaderLimits, WebLoader


async def test_file_loader_reads_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world")
    src = await FileLoader().load(str(f))
    assert src.data == b"hello world"
    assert src.mime_type == "text/plain"
    assert src.source_uri.endswith("a.txt")


async def test_file_loader_resolves_dotdot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sub = tmp_path / "sub"
    sub.mkdir()
    target = tmp_path / "real.txt"
    target.write_bytes(b"resolved")
    # A path with ".." that resolves to the real file.
    src = await FileLoader().load(str(sub / ".." / "real.txt"))
    assert src.data == b"resolved"


async def test_file_loader_missing_path() -> None:
    with pytest.raises(IngestionError) as exc_info:
        await FileLoader().load("/no/such/file.xyz")
    assert exc_info.value.code == "file_not_found"


async def test_file_loader_size_limit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 200)
    loader = FileLoader(limits=LoaderLimits(max_bytes=100))
    with pytest.raises(IngestionError) as exc_info:
        await loader.load(str(f))
    assert exc_info.value.code == "file_too_large"


async def test_web_loader_with_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"web data", headers={"content-type": "text/html; charset=utf-8"}
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        loader = WebLoader(client)
        src = await loader.load("http://example.test/page")
    assert src.data == b"web data"
    assert src.mime_type == "text/html; charset=utf-8"
    assert src.source_uri == "http://example.test/page"


async def test_web_loader_rejects_ssrf() -> None:
    from rag_core.errors import CrawlError

    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b""))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(CrawlError):
            await WebLoader(client).load("http://127.0.0.1/")
