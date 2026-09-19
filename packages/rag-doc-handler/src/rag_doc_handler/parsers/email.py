"""``.eml`` parsing via the standard library ``email`` module (RFC 822)."""

from __future__ import annotations

import asyncio
import re
from email import message_from_bytes, policy
from email.message import Message

from rag_core.documents import Document, DocumentAsset, DocumentPage
from rag_core.ids import new_id

from rag_doc_handler.parsers.base import BaseParser

__all__ = ["EmlParser"]

_HTML_TAG = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG.sub("", text)


def _parse_eml(data: bytes) -> tuple[str, str, str, str, list[str]]:
    msg: Message = message_from_bytes(data, policy=policy.default)
    subject = str(msg.get("subject", "")).strip()
    from_ = str(msg.get("from", "")).strip()
    to = str(msg.get("to", "")).strip()

    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        filename = part.get_filename()
        if filename or part.get_content_disposition() == "attachment":
            attachments.append(filename or "")
            continue
        raw = part.get_payload(decode=True)
        if isinstance(raw, bytes):
            charset = part.get_content_charset() or "utf-8"
            payload = raw.decode(charset, "replace")
        else:
            payload = ""
        if ctype == "text/plain":
            plain_parts.append(payload)
        elif ctype == "text/html":
            html_parts.append(payload)

    body = "\n".join(plain_parts) if plain_parts else _strip_html("\n".join(html_parts)).strip()
    return subject, from_, to, body, attachments


class EmlParser(BaseParser):
    """Parses ``.eml`` / RFC 822 email messages."""

    @classmethod
    def supported_types(cls) -> set[str]:
        return {".eml", "message/rfc822"}

    async def parse(self, source: str, data: bytes) -> Document:
        subject, from_, to, body, attachments = await asyncio.to_thread(_parse_eml, data)

        assets = [
            DocumentAsset(
                asset_id=str(name),
                kind="attachment",
                page_number=1,
                content_ref=str(name),
            )
            for name in attachments
            if name
        ]

        doc = Document(source_uri=source, text=body)
        if subject:
            doc.metadata.title = subject
        if from_:
            doc.metadata.custom["from"] = from_
        if to:
            doc.metadata.custom["to"] = to
        doc.pages = [DocumentPage(id=new_id(), page_number=1, text=body, blocks=[])]
        doc.assets = assets
        return self._stamp(doc, "eml")
