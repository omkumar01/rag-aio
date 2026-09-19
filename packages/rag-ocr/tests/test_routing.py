"""Tests for the escalation router."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from rag_core.errors import OCRError
from rag_core.protocols import OCRProcessor
from rag_ocr.engines import NullOCREngine, RawLine
from rag_ocr.models import OCRPageResult, OCRRegion
from rag_ocr.routing import OCRRouter, OCRRouterConfig


class FakeSemantic:
    """Records calls and returns a fixed set of regions."""

    def __init__(self, regions: list[OCRRegion] | None = None) -> None:
        self.name = "fake-vlm"
        self.calls: list[bytes] = []
        self._regions = regions or [OCRRegion(text="semantic text", confidence=0.85, kind="text")]

    async def extract(self, page_image_bytes: bytes, context: str | None = None) -> list[OCRRegion]:
        self.calls.append(page_image_bytes)
        return list(self._regions)


class FakeEngine:
    def __init__(self, name: str, lines: list[RawLine] | None = None) -> None:
        self.name = name
        self._lines = lines or []
        self.calls = 0

    async def recognize(self, image_bytes: bytes) -> list[RawLine]:
        self.calls += 1
        return list(self._lines)


def _tiny() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8, 8), color=255).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_router_is_ocr_processor_protocol() -> None:
    router = OCRRouter(engines=[NullOCREngine()], semantic=None)
    assert isinstance(router, OCRProcessor)


@pytest.mark.asyncio
async def test_zero_lines_triggers_escalation() -> None:
    semantic = FakeSemantic()
    router = OCRRouter(engines=[NullOCREngine()], semantic=semantic)
    result = await router.process_page("doc-1", 1, _tiny())
    assert len(semantic.calls) == 1
    assert len(result.regions) == 1
    assert result.regions[0].text == "semantic text"
    assert result.engine == "fake-vlm"
    assert result.mean_confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_low_confidence_triggers_escalation() -> None:
    semantic = FakeSemantic()
    low_lines = [RawLine(text="fuzzy", bbox=(0, 0, 10, 10), confidence=0.3)]
    engine = FakeEngine("low", low_lines)
    config = OCRRouterConfig(confidence_threshold=0.7, min_lines=1)
    router = OCRRouter(engines=[engine], semantic=semantic, config=config)
    result = await router.process_page("doc-1", 1, _tiny())
    assert engine.calls == 1
    assert len(semantic.calls) == 1
    assert len(result.regions) == 1


@pytest.mark.asyncio
async def test_high_confidence_no_escalation() -> None:
    semantic = FakeSemantic()
    high_lines = [RawLine(text="clear", bbox=(0, 0, 10, 10), confidence=0.95)]
    engine = FakeEngine("hi", high_lines)
    config = OCRRouterConfig(confidence_threshold=0.7, min_lines=1)
    router = OCRRouter(engines=[engine], semantic=semantic, config=config)
    result = await router.process_page("doc-1", 1, _tiny())
    assert len(semantic.calls) == 0
    assert len(result.regions) == 1
    assert result.regions[0].text == "clear"
    assert result.regions[0].confidence == 0.95
    assert result.engine == "hi"


@pytest.mark.asyncio
async def test_escalate_model_false_uses_engine_regions() -> None:
    semantic = FakeSemantic()
    low_lines = [RawLine(text="fuzzy", bbox=(0, 0, 10, 10), confidence=0.3)]
    engine = FakeEngine("low", low_lines)
    config = OCRRouterConfig(escalate_model=False, confidence_threshold=0.7)
    router = OCRRouter(engines=[engine], semantic=semantic, config=config)
    result = await router.process_page("doc-1", 1, _tiny())
    assert len(semantic.calls) == 0
    assert result.regions[0].text == "fuzzy"


@pytest.mark.asyncio
async def test_max_escalations_respected_in_document() -> None:
    semantic = FakeSemantic()
    config = OCRRouterConfig(confidence_threshold=0.7, max_escalations_per_doc=2)
    router = OCRRouter(engines=[NullOCREngine()], semantic=semantic, config=config)
    pages = [(i, _tiny()) for i in range(1, 5)]  # 4 bad pages
    result = await router.process_document("doc-x", pages)
    assert len(semantic.calls) == 2
    assert result.escalated_pages == [1, 2]
    assert len(result.pages) == 4
    # Non-escalated pages fall back to engine (null) -> empty regions.
    assert result.pages[2].regions == []
    assert result.pages[2].engine == "null"


@pytest.mark.asyncio
async def test_router_requires_at_least_one_engine() -> None:
    with pytest.raises(OCRError):
        OCRRouter(engines=[], semantic=None)  # type: ignore[arg-type]


def test_mean_confidence_property() -> None:
    single = OCRPageResult(
        page_number=0,
        regions=[OCRRegion(text="a", confidence=0.5, kind="text")],
        engine="x",
    )
    assert single.mean_confidence == pytest.approx(0.5)
    empty = OCRPageResult(page_number=0, regions=[], engine="x")
    assert empty.mean_confidence == 0.0


def test_page_result_has_width_height_language() -> None:
    page = OCRPageResult(
        page_number=3,
        regions=[],
        engine="null",
        width=100.0,
        height=200.0,
        language="en",
    )
    assert page.page_number == 3
    assert page.width == 100.0
    assert page.language == "en"
