"""End-to-end IngestionPipeline tests (no real network)."""

from __future__ import annotations

import pymupdf
import pytest
from rag_core.errors import UnsupportedFormatError
from rag_doc_handler import DedupIndex, default_pipeline
from rag_doc_handler.version import __version__


def _make_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((50, 50), "Pipeline PDF page")
    doc.set_metadata({"title": "Pipeline Doc"})
    data = doc.tobytes()
    doc.close()
    return data


async def test_pipeline_ingest_markdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "doc.md"
    f.write_text("# Hello\n\nWorld", encoding="utf-8")
    doc, is_new = await default_pipeline().ingest_file(f)
    assert is_new is True
    assert "Hello" in doc.text
    assert doc.parser_name == "text"
    assert doc.parser_version == __version__


async def test_pipeline_ingest_pdf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "doc.pdf"
    f.write_bytes(_make_pdf())
    doc, is_new = await default_pipeline().ingest_file(f)
    assert is_new is True
    assert len(doc.pages) == 1
    assert "Pipeline PDF page" in doc.pages[0].text
    assert doc.metadata.title == "Pipeline Doc"
    assert doc.parser_name == "pdf"


async def test_pipeline_ingest_raw_bytes() -> None:
    doc, is_new = await default_pipeline().ingest("note.txt", b"raw bytes payload")
    assert is_new is True
    assert "raw bytes payload" in doc.text


async def test_pipeline_dedup_returns_not_new_on_duplicate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "doc.md"
    f.write_text("same content", encoding="utf-8")
    pipeline = default_pipeline()
    dedup = DedupIndex()
    doc1, is_new1 = await pipeline.ingest_file(f, dedup=dedup)
    doc2, is_new2 = await pipeline.ingest_file(f, dedup=dedup)
    assert is_new1 is True
    assert is_new2 is False
    assert doc1.content_hash == doc2.content_hash
    assert len(dedup) == 1


async def test_pipeline_unsupported_format_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "mystery.xyz"
    f.write_bytes(b"unknown")
    with pytest.raises(UnsupportedFormatError):
        await default_pipeline().ingest_file(f)
