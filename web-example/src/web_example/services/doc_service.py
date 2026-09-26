"""Document registry service: shared CRUD over the web DB ``documents`` table.

Every ingested document (chat attachments, bulk uploads, future tracks) gets a
registry row here; this module is the single canonical layer for reading and
mutating those rows. RAG content itself (vectors, document store text) lives
outside the web DB — vector cleanup is the caller's responsibility.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import Document, User

__all__ = [
    "delete_document",
    "document_counts",
    "get_document",
    "list_documents",
    "promote_to_global",
    "register_document",
]


async def register_document(
    session: AsyncSession,
    document_id: str,
    *,
    owner_id: str | None,
    visibility: str,
    name: str,
    source_uri: str = "",
    ext: str = "",
    size_bytes: int = 0,
    num_chunks: int = 0,
    content_hash: str = "",
) -> Document:
    """Insert a registry row for an ingested document and commit it.

    ``document_id`` is the RAG document id (the id used by the vector store and
    document store), ``visibility`` is ``"user"`` or ``"global"``.
    """
    row = Document(
        id=document_id,
        owner_id=owner_id,
        visibility=visibility,
        name=name,
        source_uri=source_uri,
        ext=ext,
        size_bytes=size_bytes,
        num_chunks=num_chunks,
        content_hash=content_hash,
    )
    session.add(row)
    await session.commit()
    return row


async def list_documents(
    session: AsyncSession, user_id: str, *, include_global: bool = True
) -> list[Document]:
    """Return the user's own documents (plus global ones when asked), newest first."""
    result = await session.execute(
        text(
            "SELECT id FROM documents "
            "WHERE owner_id = :owner_id OR (:include_global = 1 AND visibility = 'global') "
            "ORDER BY created_at DESC"
        ),
        {"owner_id": user_id, "include_global": 1 if include_global else 0},
    )
    docs: list[Document] = []
    for row_id in result.scalars():
        doc = await session.get(Document, row_id)
        if doc is not None:
            docs.append(doc)
    return docs


async def get_document(session: AsyncSession, user_id: str, document_id: str) -> Document | None:
    """Fetch a registry row iff *user_id* may see it (owner, or global visibility)."""
    doc = await session.get(Document, document_id)
    if doc is None:
        return None
    if doc.owner_id == user_id or doc.visibility == "global":
        return doc
    return None


async def delete_document(
    session: AsyncSession, user_id: str, document_id: str, *, is_admin: bool = False
) -> bool:
    """Delete a registry row after a permission check.

    The owner may delete their own document; an admin may delete a global
    document (but not another user's private one). Raises
    :class:`PermissionError` when not allowed. Returns ``False`` when the row
    does not exist. Vector-store cleanup is the caller's job.
    """
    doc = await session.get(Document, document_id)
    if doc is None:
        return False
    allowed = doc.owner_id == user_id or (is_admin and doc.visibility == "global")
    if not allowed:
        msg = f"user {user_id} may not delete document {document_id}"
        raise PermissionError(msg)
    await session.delete(doc)
    await session.commit()
    return True


async def promote_to_global(session: AsyncSession, admin_user: User, document_id: str) -> bool:
    """Flip a document's visibility from ``user`` to ``global`` (admin only).

    NOTE: this only changes the registry row. Existing vectors stay in the
    owner's namespace; for retrieval from the global pool the admin should
    re-ingest the source file into the global pool.

    Returns ``False`` when the document does not exist; raises
    :class:`PermissionError` for non-admins.
    """
    if not admin_user.is_admin:
        msg = "admin access required to promote documents"
        raise PermissionError(msg)
    doc = await session.get(Document, document_id)
    if doc is None:
        return False
    if doc.visibility != "global":
        doc.visibility = "global"
        await session.commit()
    return True


async def document_counts(session: AsyncSession, user_id: str) -> dict[str, int]:
    """Return ``{"mine": <owned rows>, "global": <global rows>}`` for *user_id*."""
    mine_result = await session.execute(
        text("SELECT COUNT(*) FROM documents WHERE owner_id = :owner_id"),
        {"owner_id": user_id},
    )
    global_result = await session.execute(
        text("SELECT COUNT(*) FROM documents WHERE visibility = 'global'"),
        {},
    )
    return {
        "mine": int(mine_result.scalar_one()),
        "global": int(global_result.scalar_one()),
    }
