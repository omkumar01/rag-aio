"""Chat page, conversation, and streaming ask routes (SSE over HTMX/fetch)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import UploadFile

from web_example.auth.deps import get_current_user
from web_example.db import Conversation, Document, User
from web_example.services import chat_service
from web_example.services.rag_factory import RagManager

__all__ = ["router"]

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

MAX_ATTACHMENTS_PER_TYPE = 5
DOCUMENT_EXTS = frozenset(
    {"pdf", "txt", "md", "markdown", "html", "htm", "docx", "xlsx", "pptx", "eml"}
)
IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "webp"})


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _uploads(form: Any, key: str) -> list[UploadFile]:
    """File inputs under *key*, ignoring the empty entries browsers send."""
    return [item for item in form.getlist(key) if isinstance(item, UploadFile) and item.filename]


def _ext_of(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


async def _sse_events(events: Sequence[dict[str, Any]]) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        for event in events:
            yield _sse(event)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@router.get("/chat")
async def chat_page(request: Request, user: Annotated[User, Depends(get_current_user)]) -> Response:
    return cast(Response, request.app.state.templates.TemplateResponse(request, "chat.html", {}))


@router.get("/chat/conversations")
async def conversations_partial(
    request: Request, user: Annotated[User, Depends(get_current_user)]
) -> Response:
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        conversations = await chat_service.list_conversations(session, user.id)
    return cast(
        Response,
        request.app.state.templates.TemplateResponse(
            request, "partials/chat_sidebar.html", {"conversations": conversations}
        ),
    )


@router.get("/chat/conversations/{conversation_id}/messages")
async def messages_partial(
    conversation_id: str, request: Request, user: Annotated[User, Depends(get_current_user)]
) -> Response:
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        messages = await chat_service.get_messages(session, conversation_id, user.id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return cast(
        Response,
        request.app.state.templates.TemplateResponse(
            request,
            "partials/chat_thread.html",
            {"messages": messages, "conversation_id": conversation_id},
        ),
    )


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, request: Request, user: Annotated[User, Depends(get_current_user)]
) -> Response:
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :conversation_id"),
            {"conversation_id": conversation_id},
        )
        await session.delete(conversation)
        await session.commit()
    return Response(status_code=200)


@router.post("/chat/ask")
async def ask(request: Request, user: Annotated[User, Depends(get_current_user)]) -> Response:
    manager: RagManager | None = getattr(request.app.state, "rag", None)
    if manager is None:
        return await _sse_events(
            [{"type": "error", "message": "RAG engine not initialised."}, {"type": "done"}]
        )

    form = await request.form()
    message = str(form.get("message", "")).strip()
    doc_files = _uploads(form, "doc_files")
    image_files = _uploads(form, "image_files")

    if len(doc_files) > MAX_ATTACHMENTS_PER_TYPE or len(image_files) > MAX_ATTACHMENTS_PER_TYPE:
        return JSONResponse(
            status_code=413,
            content={"error": f"Too many attachments (max {MAX_ATTACHMENTS_PER_TYPE} per type)."},
        )
    for upload in doc_files:
        if _ext_of(upload.filename or "") not in DOCUMENT_EXTS:
            return JSONResponse(
                status_code=415, content={"error": f"Unsupported document type: {upload.filename}"}
            )
    for upload in image_files:
        if _ext_of(upload.filename or "") not in IMAGE_EXTS:
            return JSONResponse(
                status_code=415, content={"error": f"Unsupported image type: {upload.filename}"}
            )
    if not message and not doc_files and not image_files:
        return JSONResponse(status_code=422, content={"error": "Please enter a message."})

    uploads: list[tuple[str, bytes]] = []
    attachments_meta: list[dict[str, Any]] = []
    for upload in [*doc_files, *image_files]:
        data = await upload.read()
        name = upload.filename or "file"
        uploads.append((name, data))
        attachments_meta.append({"name": name, "size": len(data)})

    conversation_id = str(form.get("conversation_id") or "") or None

    async def event_stream() -> AsyncIterator[str]:
        maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
        async with maker() as session:
            try:
                await _ingest_attachments(session, manager, user, uploads)
            except Exception as exc:
                logger.exception("attachment ingestion failed")
                yield _sse({"type": "error", "message": f"Ingestion failed: {exc}"})
                yield _sse({"type": "done"})
                return
            try:
                async for event in chat_service.stream_chat(
                    session, manager, user, conversation_id, message, attachments_meta
                ):
                    yield _sse(event)
            except LookupError:
                yield _sse({"type": "error", "message": "Conversation not found."})
                yield _sse({"type": "done"})
            except Exception as exc:
                logger.exception("chat stream failed")
                yield _sse({"type": "error", "message": str(exc)})
                yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _ingest_attachments(
    session: Any, manager: RagManager, user: User, uploads: Sequence[tuple[str, bytes]]
) -> None:
    """Ingest each attachment into the user pool and register it."""
    for name, data in uploads:
        before = await manager.count_chunks(session, user.id)
        rag_doc = await manager.ingest_bytes(session, user.id, name, data)
        after = await manager.count_chunks(session, user.id)
        session.add(
            Document(
                id=rag_doc.id,
                owner_id=user.id,
                visibility="user",
                name=name,
                source_uri=rag_doc.source_uri,
                ext=_ext_of(name),
                size_bytes=len(data),
                num_chunks=max(after - before, 0),
                content_hash=rag_doc.content_hash,
            )
        )
        await session.commit()
