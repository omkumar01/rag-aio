"""Tests for the OCR pipeline and the regions->blocks projection."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from rag_core.documents import BoundingBox
from rag_ocr.engines import NullOCREngine
from rag_ocr.models import OCRPageResult, OCRRegion
from rag_ocr.pipeline import OCRPipeline, regions_to_blocks
from rag_ocr.routing import OCRRouter, OCRRouterConfig


def _img(page_number: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (16, 16), color=((page_number * 40) % 256)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeSemantic:
    name = "fake-vlm"

    async def extract(self, page_image_bytes: bytes, context: str | None = None) -> list[OCRRegion]:
        return [OCRRegion(text="region", confidence=0.85, kind="text")]


@pytest.mark.asyncio
async def test_pipeline_processes_two_images_in_order() -> None:
    router = OCRRouter(
        engines=[NullOCREngine()],
        semantic=_FakeSemantic(),
        config=OCRRouterConfig(confidence_threshold=0.7, max_escalations_per_doc=5),
    )
    pipeline = OCRPipeline(router, preprocess_enabled=True, max_pages=200)
    result = await pipeline.process_images([(0, _img(0)), (1, _img(1))], document_id="doc-1")
    assert result.document_id == "doc-1"
    assert [p.page_number for p in result.pages] == [0, 1]
    assert result.escalated_pages == [0, 1]
    assert len(result.pages) == 2


@pytest.mark.asyncio
async def test_pipeline_enforces_max_pages() -> None:
    router = OCRRouter(engines=[NullOCREngine()], semantic=None)
    pipeline = OCRPipeline(router, preprocess_enabled=False, max_pages=2)
    result = await pipeline.process_images([(i, _img(i)) for i in range(5)])
    assert len(result.pages) == 2
    assert [p.page_number for p in result.pages] == [0, 1]


@pytest.mark.asyncio
async def test_pipeline_generates_document_id() -> None:
    router = OCRRouter(engines=[NullOCREngine()], semantic=None)
    pipeline = OCRPipeline(router, preprocess_enabled=False)
    result = await pipeline.process_images([(0, _img(0))])
    assert result.document_id  # non-empty


def test_regions_to_blocks_preserves_fields() -> None:
    page = OCRPageResult(
        page_number=7,
        regions=[
            OCRRegion(text="hello", confidence=0.9, kind="text", bbox=None),
            OCRRegion(
                text="world",
                confidence=0.6,
                kind="heading",
                bbox=BoundingBox(x0=1, y0=2, x1=3, y1=4),
            ),
        ],
        engine="null",
    )
    blocks = regions_to_blocks(page, page_id="page-7")
    assert len(blocks) == 2
    assert blocks[0].page_id == "page-7"
    assert blocks[0].text == "hello"
    assert blocks[0].kind == "text"
    assert blocks[0].confidence == 0.9
    assert blocks[0].order == 0
    assert blocks[0].id  # non-empty id
    assert blocks[1].kind == "heading"
    assert blocks[1].bbox is not None
    assert blocks[1].bbox.x0 == 1
    assert blocks[1].bbox.y1 == 4
    assert blocks[1].order == 1


def test_regions_to_blocks_empty() -> None:
    page = OCRPageResult(page_number=1, regions=[], engine="null")
    assert regions_to_blocks(page, "p1") == []
