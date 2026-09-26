"""Document explorer routes: listing, preview, delete, promote, JSON API."""

from __future__ import annotations

import logging
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.auth.deps import get_current_user, require_admin
from web_example.db import Document, User
from web_example.services import doc_service
from web_example.services.rag_factory import RagManager

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

#: Maximum characters of document text shown in the preview page.
_PREVIEW_LIMIT = 800


@router.get("/documents")
async def documents_page(
    request: Request, user: Annotated[User, Depends(get_current_user)]
) -> Response:
    """Document explorer page; HTMX requests get only the table partial."""
    if request.headers.get("HX-Request"):
        return cast(Response, await _table_partial(request, user))
    return cast(
        Response,
        request.app.state.templates.TemplateResponse(
            request,
            "documents.html",
            await _table_context(request, user),
        ),
    )


@router.get("/documents/{document_id}/preview")
async def document_preview(
    request: Request, document_id: str, user: Annotated[User, Depends(get_current_user)]
) -> Response:
    """Metadata page with a best-effort short text preview (never 500s)."""
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        doc = await doc_service.get_document(session, user.id, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        preview_text = await _fetch_preview(request, session, user, doc)
    return cast(
        Response,
        request.app.state.templates.TemplateResponse(
            request,
            "doc_preview.html",
            {
                "doc": _doc_view(doc, user),
                "source_uri": doc.source_uri,
                "content_hash": doc.content_hash,
                "preview_text": preview_text,
            },
        ),
    )


@router.post("/documents/{document_id}/delete")
async def document_delete(
    request: Request, document_id: str, user: Annotated[User, Depends(get_current_user)]
) -> Response:
    """Permission-checked registry delete plus best-effort vector cleanup."""
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        doc = await doc_service.get_document(session, user.id, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        visibility = doc.visibility
        try:
            await doc_service.delete_document(session, user.id, document_id, is_admin=user.is_admin)
        except PermissionError:
            raise HTTPException(
                status_code=403, detail="Not allowed to delete this document."
            ) from None
        manager: RagManager | None = getattr(request.app.state, "rag", None)
        if manager is not None:
            try:
                await manager.delete_document(
                    session,
                    user.id,
                    document_id,
                    global_pool=(visibility == "global"),
                )
            except Exception:
                logger.warning("vector cleanup failed for document %s", document_id)
    if request.headers.get("HX-Request"):
        return cast(Response, await _table_partial(request, user))
    return RedirectResponse("/documents", status_code=303)


@router.post("/documents/{document_id}/promote")
async def document_promote(
    request: Request, document_id: str, user: Annotated[User, Depends(require_admin)]
) -> Response:
    """Admin-only: flip a registry row to global visibility."""
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        promoted = await doc_service.promote_to_global(session, user, document_id)
        if not promoted:
            raise HTTPException(status_code=404, detail="Document not found.")
    if request.headers.get("HX-Request"):
        return cast(Response, await _table_partial(request, user))
    return RedirectResponse("/documents", status_code=303)


@router.get("/documents/api/list")
async def documents_api_list(
    request: Request, user: Annotated[User, Depends(get_current_user)]
) -> list[dict[str, Any]]:
    """JSON list of the documents visible to the current user."""
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        docs = await doc_service.list_documents(session, user.id)
    return [
        {
            "id": doc.id,
            "name": doc.name,
            "visibility": doc.visibility,
            "ext": doc.ext,
            "size": doc.size_bytes,
            "chunks": doc.num_chunks,
            "created_at": doc.created_at.isoformat(),
        }
        for doc in docs
    ]


# -- helpers -----------------------------------------------------------------


async def _table_context(request: Request, user: User) -> dict[str, Any]:
    """Shared context for the documents page and its HTMX partial."""
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        docs = await doc_service.list_documents(session, user.id)
        counts = await doc_service.document_counts(session, user.id)
    mine_docs = [_doc_view(doc, user) for doc in docs if doc.owner_id == user.id]
    global_docs = [_doc_view(doc, user) for doc in docs if doc.visibility == "global"]
    return {
        "user": user,
        "counts": counts,
        "mine_docs": mine_docs,
        "global_docs": global_docs,
    }


async def _table_partial(request: Request, user: User) -> Any:
    """Render only the table partial (used for HTMX refreshes)."""
    return request.app.state.templates.TemplateResponse(
        request, "partials/doc_table.html", await _table_context(request, user)
    )


def _doc_view(doc: Document, user: User) -> dict[str, Any]:
    """Template-friendly view dict for one registry row."""
    return {
        "id": doc.id,
        "name": doc.name,
        "ext": doc.ext,
        "size": _human_size(doc.size_bytes),
        "num_chunks": doc.num_chunks,
        "visibility": doc.visibility,
        "created_at": doc.created_at.strftime("%Y-%m-%d %H:%M"),
        "can_delete": doc.owner_id == user.id or (user.is_admin and doc.visibility == "global"),
        "can_promote": user.is_admin and doc.visibility == "user",
    }


async def _fetch_preview(
    request: Request, session: AsyncSession, user: User, doc: Document
) -> str | None:
    """Best-effort short text preview from the RAG document store (never raises)."""
    try:
        manager: RagManager = request.app.state.rag
        rag = (
            await manager.get_global(session)
            if doc.visibility == "global"
            else await manager.get(session, user.id)
        )
        store = getattr(getattr(rag, "services", None), "document_store", None)
        getter = getattr(store, "get", None)
        if not callable(getter):
            return None
        rag_doc = await getter(doc.id)
        if rag_doc is None:
            return None
        text_content = str(getattr(rag_doc, "text", "") or "").strip()
        if not text_content:
            return None
        if len(text_content) > _PREVIEW_LIMIT:
            text_content = text_content[: _PREVIEW_LIMIT - 3] + "..."
        return text_content
    except Exception:
        logger.info("text preview unavailable for document %s", doc.id)
        return None


def _human_size(size_bytes: int) -> str:
    """Render a byte count as a short human-readable string."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            whole = f"{value:.0f}" if unit == "B" else f"{value:.1f}"
            return f"{whole} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
