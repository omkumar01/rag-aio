"""Chat persistence + streaming glue between the HTTP routes and the RAG engine.

Event dicts yielded by :func:`stream_chat` (``meta`` → ``delta``… → ``done``)
are rendered as SSE by the routes layer; persistence happens around them.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from rag_core import ProviderError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import Conversation, Message, User
from web_example.services.rag_factory import RagManager

if TYPE_CHECKING:
    from rag_orchestrator import AskResult

__all__ = ["get_messages", "list_conversations", "stream_chat"]

_TITLE_MAX_CHARS = 60


def _title_for(query: str) -> str:
    return query.strip()[:_TITLE_MAX_CHARS] or "New chat"


async def stream_chat(
    session: AsyncSession,
    manager: RagManager,
    user: User,
    conversation_id: str | None,
    query: str,
    attachments: Sequence[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Create/continue a conversation and stream one assistant turn.

    Yields ``{"type": "meta", ...}`` first, then ``{"type": "delta", ...}``
    events, and finally ``{"type": "done"}`` once the assistant message is
    persisted. Raises :class:`LookupError` when *conversation_id* points at a
    conversation the user does not own.
    """
    conversation: Conversation | None = None
    if conversation_id:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user.id:
            msg = "conversation not found"
            raise LookupError(msg)
    if conversation is None:
        conversation = Conversation(user_id=user.id, title=_title_for(query))
        session.add(conversation)
        await session.flush()

    session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=query,
            attachments=[
                {"name": att.get("name", ""), "size": att.get("size", 0)} for att in attachments
            ],
        )
    )
    await session.commit()

    yield {"type": "meta", "conversation_id": conversation.id, "title": conversation.title}

    rag = await manager.get(session, user.id)
    parts: list[str] = []
    try:
        generator = rag.services.generator
        if generator is not None and hasattr(generator, "stream"):
            stream = cast(AsyncIterator[str], await rag.ask(query, stream=True))
            async for delta in stream:
                parts.append(delta)
                yield {"type": "delta", "text": delta}
        else:
            result = cast("AskResult", await rag.ask(query, stream=False))
            parts.append(result.answer)
            yield {"type": "delta", "text": result.answer}
    except Exception as exc:
        yield {"type": "error", "message": _friendly_error(exc)}
        return

    session.add(Message(conversation_id=conversation.id, role="assistant", content="".join(parts)))
    await session.commit()
    yield {"type": "done"}


def _friendly_error(exc: Exception) -> str:
    """A user-presentable message for a failed generation turn.

    The orchestrator wraps low-level failures (e.g. a numpy "shapes not
    aligned" error from a dimension mismatch) into generic RetrievalError
    text, so the whole exception chain is inspected for known signatures.
    """
    chain_text = " | ".join(str(item) for item in _error_chain(exc)).lower()
    if "not aligned" in chain_text or "dimension" in chain_text:
        return (
            "The stored vectors do not match the current embedding model "
            "(dimension mismatch). Re-ingest your documents after changing "
            "the embedding model or dimension."
        )
    if isinstance(exc, ProviderError):
        return (
            "The chat model endpoint is unreachable or returned an error. "
            "Check the generation settings (provider base URL / model) and "
            "that the server is running."
        )
    return _clip_error(str(exc))


def _error_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* and every linked cause/context, cycle-safe."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _clip_error(text: str) -> str:
    return text if len(text) <= 300 else text[:297] + "..."


async def list_conversations(session: AsyncSession, user_id: str) -> list[dict[str, Any]]:
    """The user's conversations, newest first."""
    rows = (
        (
            await session.execute(
                text(
                    "SELECT id, title, updated_at FROM conversations"
                    " WHERE user_id = :user_id"
                    " ORDER BY updated_at DESC, created_at DESC"
                    " LIMIT 200"
                ),
                {"user_id": user_id},
            )
        )
        .mappings()
        .all()
    )
    return [dict(row) for row in rows]


async def get_messages(
    session: AsyncSession, conversation_id: str, user_id: str
) -> list[dict[str, Any]] | None:
    """Messages of a conversation, or ``None`` when missing / not owned."""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        return None
    rows = (
        (
            await session.execute(
                text(
                    "SELECT id, role, content, attachments, created_at FROM messages"
                    " WHERE conversation_id = :conversation_id"
                    " ORDER BY created_at ASC, id ASC"
                ),
                {"conversation_id": conversation_id},
            )
        )
        .mappings()
        .all()
    )
    messages: list[dict[str, Any]] = []
    for row in rows:
        message = dict(row)
        message["attachments"] = _decode_attachments(message.get("attachments"))
        messages.append(message)
    return messages


def _decode_attachments(raw: Any) -> list[dict[str, Any]]:
    """Raw SQL bypasses the ORM JSON type; decode string payloads here."""
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return raw if isinstance(raw, list) else []
