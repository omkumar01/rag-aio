"""Tests for the VLM semantic extractor."""

from __future__ import annotations

import base64
import os

import httpx
import pytest
from rag_core.errors import OCRError
from rag_ocr.semantic import VLMSemanticExtractor, classify_kind


def _canned_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _transport(content: str) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, json=_canned_response(content)))


@pytest.mark.asyncio
async def test_extract_table_classification() -> None:
    extractor = VLMSemanticExtractor(transport=_transport("| a | b |\n| --- | --- |\n| 1 | 2 |"))
    regions = await extractor.extract(b"\x89PNG\r\n\x1a\n")
    assert len(regions) == 1
    assert regions[0].kind == "table"
    assert "| a | b |" in regions[0].text
    assert regions[0].confidence == 0.85
    assert regions[0].bbox is None
    await extractor.aclose()


@pytest.mark.asyncio
async def test_extract_heading_classification() -> None:
    extractor = VLMSemanticExtractor(transport=_transport("# A Heading\n\nsome text"))
    regions = await extractor.extract(b"\x89PNG\r\n\x1a\n")
    assert regions[0].kind == "heading"
    assert regions[0].text.startswith("#")
    await extractor.aclose()


@pytest.mark.asyncio
async def test_extract_text_classification() -> None:
    extractor = VLMSemanticExtractor(transport=_transport("Plain body text without markup."))
    regions = await extractor.extract(b"\x89PNG\r\n\x1a\n")
    assert regions[0].kind == "text"
    await extractor.aclose()


@pytest.mark.asyncio
async def test_classify_kind_dispatch() -> None:
    assert classify_kind("| x | y |\n|---|---|") == "table"
    assert classify_kind("# Title") == "heading"
    assert classify_kind("Just text") == "text"


@pytest.mark.asyncio
async def test_api_key_resolved_from_env_and_sent() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization", "")
        return httpx.Response(200, json=_canned_response("hello"))

    os.environ["RAG_OCR_VLM_KEY"] = "secret-token-123"
    try:
        extractor = VLMSemanticExtractor(
            api_key_ref="RAG_OCR_VLM_KEY", transport=httpx.MockTransport(handler)
        )
        regions = await extractor.extract(b"\x89PNG\r\n\x1a\n")
    finally:
        del os.environ["RAG_OCR_VLM_KEY"]

    assert captured["authorization"] == "Bearer secret-token-123"
    # The key must never leak into the produced region.
    assert "secret-token-123" not in regions[0].text
    await extractor.aclose()


@pytest.mark.asyncio
async def test_timeout_maps_to_ocr_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated")

    extractor = VLMSemanticExtractor(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OCRError):
            await extractor.extract(b"\x89PNG\r\n\x1a\n")
    finally:
        await extractor.aclose()


@pytest.mark.asyncio
async def test_image_sent_as_data_url() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8", errors="replace")
        captured.append(body)
        return httpx.Response(200, json=_canned_response("ok"))

    payload = b"\x89PNG\r\n\x1a\nfakedata"
    extractor = VLMSemanticExtractor(transport=httpx.MockTransport(handler))
    await extractor.extract(payload)
    await extractor.aclose()

    assert len(captured) == 1
    assert "data:image/png;base64," in captured[0]
    assert base64.b64encode(payload).decode("ascii") in captured[0]


@pytest.mark.asyncio
async def test_server_error_maps_to_ocr_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    extractor = VLMSemanticExtractor(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OCRError):
            await extractor.extract(b"\x89PNG\r\n\x1a\n")
    finally:
        await extractor.aclose()
