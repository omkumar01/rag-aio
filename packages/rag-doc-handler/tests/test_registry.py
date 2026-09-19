"""Tests for the parser registry and install-hint behaviour."""

from __future__ import annotations

import pytest
from rag_core.documents import Document
from rag_core.errors import UnsupportedFormatError
from rag_doc_handler.parsers.base import BaseParser, ParserRegistry, install_hint_for
from rag_doc_handler.parsers.text import TextParser


def test_install_hint_for_optional_formats() -> None:
    assert "msg" in install_hint_for(".msg")
    assert "rtf" in install_hint_for(".rtf")
    assert "docling" in install_hint_for(".doc")
    assert "docling" in install_hint_for(".xls")


def test_install_hint_unknown_is_none() -> None:
    assert install_hint_for(".pdf") is None
    assert install_hint_for(".xyz") is None
    assert install_hint_for("application/json") is None


def test_registry_unknown_extension_raises_with_hint() -> None:
    reg = ParserRegistry()
    with pytest.raises(UnsupportedFormatError) as exc_info:
        reg.get(".rtf")
    assert exc_info.value.code == "unsupported_format"
    assert "install_hint" in exc_info.value.details
    assert "rtf" in exc_info.value.details["install_hint"]


def test_registry_unknown_core_extension_raises_without_hint() -> None:
    reg = ParserRegistry()
    with pytest.raises(UnsupportedFormatError) as exc_info:
        reg.get(".xyz")
    assert "install_hint" not in exc_info.value.details


def test_registry_get_or_none_returns_none() -> None:
    reg = ParserRegistry()
    assert reg.get_or_none(".missing") is None


def test_registry_register_instance_and_get() -> None:
    reg = ParserRegistry()
    text_parser = TextParser()
    reg.register(text_parser)
    assert ".txt" in reg.get(".txt").supported_types()
    assert reg.get(".md").supported_types() == {".txt", ".md", ".markdown"}


def test_registry_register_as_decorator() -> None:
    reg = ParserRegistry()

    @reg.register
    class DummyParser(BaseParser):
        @classmethod
        def supported_types(cls) -> set[str]:
            return {".zzz"}

        async def parse(self, source: str, data: bytes) -> Document:
            return Document(source_uri=source, text="dummy")

    assert reg.get(".zzz").supported_types() == {".zzz"}


def test_default_registry_has_core_types() -> None:
    from rag_doc_handler import default_registry

    registry = default_registry()
    types = registry.registered_types()
    for ext in [".txt", ".md", ".html", ".htm", ".pdf", ".docx", ".xlsx", ".pptx", ".eml"]:
        assert ext in types
