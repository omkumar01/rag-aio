"""HTML parsing with BeautifulSoup + lxml."""

from __future__ import annotations

import asyncio
import mimetypes

from bs4 import BeautifulSoup
from bs4.element import Tag
from rag_core.documents import BlockKind, Document, DocumentAsset, DocumentPage, PageBlock
from rag_core.ids import new_id

from rag_doc_handler.parsers.base import BaseParser

__all__ = ["HTMLParser"]

_MAX_LINKS = 20
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Tags whose rendered text should become a single DocumentPage block. The walker emits
# one block per matching element and does NOT recurse into it, preventing
# double-counting from nested containers.
_BLOCK_TAGS = _HEADINGS | {"p", "li", "pre", "code", "table", "blockquote", "div", "span"}


def _block_kind(name: str) -> BlockKind:
    if name in _HEADINGS:
        return "heading"
    if name == "table":
        return "table"
    if name == "li":
        return "list"
    if name in {"pre", "code"}:
        return "other"
    return "text"


def _table_text(table: Tag) -> str:
    """Render a ``<table>`` as `` | ``-joined cells per row, newline per row."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        joined = " | ".join(c for c in cells if c)
        if joined:
            rows.append(joined)
    return "\n".join(rows)


def _walk_blocks(tag: Tag, blocks: list[PageBlock]) -> None:
    """Emit one block per block-level descendant of *tag*, in document order."""
    for child in tag.children:
        if isinstance(child, Tag):
            name = child.name.lower() if isinstance(child.name, str) else ""
            if name in _BLOCK_TAGS:
                text = _table_text(child) if name == "table" else child.get_text(" ", strip=True)
                if text:
                    blocks.append(
                        PageBlock(
                            id=new_id(),
                            page_id="",
                            kind=_block_kind(name),
                            text=text,
                            order=len(blocks),
                        )
                    )
            else:
                _walk_blocks(child, blocks)
        # NavigableString children are ignored; their text is captured by ancestor blocks.


class HTMLParser(BaseParser):
    """Parses ``.html`` / ``.htm`` documents into a single :class:`DocumentPage`."""

    @classmethod
    def supported_types(cls) -> set[str]:
        return {".html", ".htm"}

    async def parse(self, source: str, data: bytes) -> Document:
        return await asyncio.to_thread(self._parse_sync, source, data)

    def _parse_sync(self, source: str, data: bytes) -> Document:
        soup = BeautifulSoup(data, "lxml")
        # Strip scripting / styling before any text extraction.
        for junk in soup("script"):
            junk.decompose()
        for junk in soup("style"):
            junk.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else None

        blocks: list[PageBlock] = []
        _walk_blocks(soup, blocks)
        page_text = "\n\n".join(b.text for b in blocks)

        # Image assets.
        assets: list[DocumentAsset] = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            mime, _ = mimetypes.guess_type(str(src))
            assets.append(
                DocumentAsset(
                    asset_id=str(src),
                    kind="image",
                    page_number=1,
                    content_ref=str(src),
                    mime_type=mime,
                )
            )

        # Top-N outbound links carried in metadata for provenance.
        links = [str(a["href"]) for a in soup.find_all("a", href=True) if a.get("href")][
            :_MAX_LINKS
        ]

        page = DocumentPage(id=new_id(), page_number=1, text=page_text, blocks=blocks)
        doc = Document(source_uri=source, text=page_text)
        doc.metadata.title = title
        doc.pages = [page]
        doc.assets = assets
        if links:
            doc.metadata.custom["links"] = "\n".join(links)
        return self._stamp(doc, "html")
