"""Tests for rag_doc_handler.detection."""

from __future__ import annotations

import pytest
from rag_doc_handler.detection import detect_extension, guess_mime, sniff_mime


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("report.pdf", ".pdf"),
        ("REPORT.PDF", ".pdf"),
        ("/tmp/doc.PDF", ".pdf"),
        ("C:\\docs\\file.MD", ".md"),
        ("page.markdown", ".markdown"),
        ("http://x.host/a/b.txt?foo=bar#frag", ".txt"),
        ("file:///c/docs/notes.MD", ".md"),
        ("noext", ""),
        (".bashrc", ""),
        ("/", ""),
        ("archive.tar.gz", ".gz"),
    ],
)
def test_detect_extension(uri: str, expected: str) -> None:
    assert detect_extension(uri) == expected


def test_sniff_mime_pdf() -> None:
    assert sniff_mime(b"%PDF-1.4\nstuff") == "application/pdf"


def test_sniff_mime_zip() -> None:
    assert sniff_mime(b"PK\x03\x04\x14\x00" + b"\x00" * 20) == "application/zip"


@pytest.mark.parametrize(
    "data",
    [
        b"<html><body>hi</body></html>",
        b"   <HTML>\n<head></head>",
        b"<!DOCTYPE html><html>",
        b"\n\t<Html> ",
    ],
)
def test_sniff_mime_html(data: bytes) -> None:
    assert sniff_mime(data) == "text/html"


def test_sniff_mime_json() -> None:
    assert sniff_mime(b'{"key": "value"}') == "application/json"
    assert sniff_mime(b'   \n  {"a": 1}') == "application/json"


def test_sniff_mime_none() -> None:
    assert sniff_mime(b"plain text without magic") is None
    assert sniff_mime(b"") is None


def test_guess_mime_prefers_sniff() -> None:
    # Even though .txt would guess text/plain, sniff wins when PDF magic present.
    assert guess_mime("doc.pdf", b"%PDF-1.4") == "application/pdf"


def test_guess_mime_falls_back_to_extension() -> None:
    assert guess_mime("doc.pdf") == "application/pdf"
    assert guess_mime("page.html") == "text/html"


def test_guess_mime_unknown() -> None:
    assert guess_mime("weird.xyz") is None
