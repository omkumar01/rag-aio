"""Service- and route-level tests for the document explorer."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import Document, User
from web_example.services.doc_service import (
    delete_document,
    document_counts,
    get_document,
    list_documents,
    promote_to_global,
    register_document,
)


async def _user_id(session: AsyncSession, username: str) -> str:
    result = await session.execute(
        text("SELECT id FROM users WHERE username = :username"), {"username": username}
    )
    return str(result.scalar_one())


async def _add_user(session: AsyncSession, username: str, *, is_admin: bool = False) -> User:
    user = User(id=f"u-{username}", username=username, password_hash="h", is_admin=is_admin)
    session.add(user)
    await session.commit()
    return user


async def _add_doc(
    session: AsyncSession,
    doc_id: str,
    name: str,
    *,
    owner_id: str | None = None,
    visibility: str = "user",
) -> Document:
    return await register_document(
        session,
        doc_id,
        owner_id=owner_id,
        visibility=visibility,
        name=name,
        source_uri=f"file://{name}",
        ext="txt",
        size_bytes=len(name) * 10,
        num_chunks=2,
        content_hash=f"hash-{doc_id}",
    )


# -- service level -----------------------------------------------------------


async def test_register_and_list_separation(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    await _add_user(session, "bob")
    alice_id = await _user_id(session, "alice")
    bob_id = await _user_id(session, "bob")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)
    await _add_doc(session, "doc-b", "bob-doc.txt", owner_id=bob_id)
    await _add_doc(session, "doc-g", "global-doc.txt", visibility="global")

    mine_only = await list_documents(session, alice_id, include_global=False)
    assert [doc.id for doc in mine_only] == ["doc-a"]

    visible = await list_documents(session, alice_id)
    assert {doc.id for doc in visible} == {"doc-a", "doc-g"}


async def test_get_document_visibility_rules(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    await _add_user(session, "bob")
    alice_id = await _user_id(session, "alice")
    bob_id = await _user_id(session, "bob")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)
    await _add_doc(session, "doc-g", "global-doc.txt", visibility="global")

    assert await get_document(session, alice_id, "doc-a") is not None
    assert await get_document(session, bob_id, "doc-a") is None
    assert await get_document(session, bob_id, "doc-g") is not None
    assert await get_document(session, alice_id, "missing") is None


async def test_delete_owner_ok_other_user_forbidden(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    await _add_user(session, "bob")
    alice_id = await _user_id(session, "alice")
    bob_id = await _user_id(session, "bob")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)

    assert await delete_document(session, alice_id, "doc-a") is True
    assert await session.get(Document, "doc-a") is None

    await _add_doc(session, "doc-a2", "alice-doc-2.txt", owner_id=alice_id)
    with pytest.raises(PermissionError):
        await delete_document(session, bob_id, "doc-a2")
    assert await session.get(Document, "doc-a2") is not None


async def test_admin_can_delete_global_doc(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    await _add_user(session, "admin", is_admin=True)
    admin_id = await _user_id(session, "admin")
    await _add_doc(session, "doc-g", "global-doc.txt", visibility="global")

    assert await delete_document(session, admin_id, "doc-g", is_admin=True) is True
    assert await session.get(Document, "doc-g") is None


async def test_admin_cannot_delete_other_users_private_doc(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    await _add_user(session, "admin", is_admin=True)
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)

    with pytest.raises(PermissionError):
        await delete_document(session, "u-admin", "doc-a", is_admin=True)


async def test_promote_admin_only(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    await _add_user(session, "admin", is_admin=True)
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)

    admin_user = await session.get(User, "u-admin")
    assert admin_user is not None
    with pytest.raises(PermissionError):
        await promote_to_global(session, await session.get(User, "u-alice"), "doc-a")

    assert await promote_to_global(session, admin_user, "doc-a") is True
    promoted = await session.get(Document, "doc-a")
    assert promoted is not None
    assert promoted.visibility == "global"


async def test_document_counts(session: AsyncSession) -> None:
    await _add_user(session, "alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)
    await _add_doc(session, "doc-g1", "global-1.txt", visibility="global")
    await _add_doc(session, "doc-g2", "global-2.txt", visibility="global")

    counts = await document_counts(session, alice_id)
    assert counts == {"mine": 1, "global": 2}


# -- route level -------------------------------------------------------------


async def test_documents_page_redirects_anonymous(client: AsyncClient) -> None:
    response = await client.get("/documents", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


async def test_documents_page_lists_my_docs_and_global(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-private-doc.txt", owner_id=alice_id)
    await _add_doc(session, "doc-g", "shared-global-doc.txt", visibility="global")

    response = await client.get("/documents")
    assert response.status_code == 200
    assert "alice-private-doc.txt" in response.text
    assert "shared-global-doc.txt" in response.text
    assert "global" in response.text  # visibility badge


async def test_documents_page_htmx_returns_partial_only(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)

    response = await client.get("/documents", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "alice-doc.txt" in response.text
    assert "<html" not in response.text.lower()


async def test_preview_shows_metadata_without_500(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "preview-me.txt", owner_id=alice_id)

    response = await client.get("/documents/doc-a/preview")
    assert response.status_code == 200
    assert "preview-me.txt" in response.text
    assert "doc-a" in response.text or "hash-doc-a" in response.text


async def test_delete_via_route_removes_row(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "delete-me.txt", owner_id=alice_id)

    response = await client.post("/documents/doc-a/delete")
    assert response.status_code in (200, 303)
    assert await session.get(Document, "doc-a") is None

    listing = await client.get("/documents")
    assert "delete-me.txt" not in listing.text


async def test_non_admin_promote_returns_403(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")  # first user becomes admin
    await client.post("/logout")
    await register("bob")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)

    response = await client.post("/documents/doc-a/promote")
    assert response.status_code == 403


async def test_admin_promote_via_route(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")  # first user becomes admin
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-doc.txt", owner_id=alice_id)

    response = await client.post("/documents/doc-a/promote", follow_redirects=False)
    assert response.status_code in (200, 303)
    doc = await session.get(Document, "doc-a")
    assert doc is not None
    assert doc.visibility == "global"


async def test_user_b_cannot_see_user_a_docs(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "alice-secret-doc.txt", owner_id=alice_id)

    await client.post("/logout")
    await register("bob")

    listing = await client.get("/documents")
    assert listing.status_code == 200
    assert "alice-secret-doc.txt" not in listing.text

    preview = await client.get("/documents/doc-a/preview", follow_redirects=False)
    assert preview.status_code in (303, 404)

    delete_attempt = await client.post("/documents/doc-a/delete", follow_redirects=False)
    assert delete_attempt.status_code in (303, 403, 404)


async def test_api_list_returns_json(
    client: AsyncClient, register: Any, session: AsyncSession
) -> None:
    await register("alice")
    alice_id = await _user_id(session, "alice")
    await _add_doc(session, "doc-a", "api-doc.txt", owner_id=alice_id)
    await _add_doc(session, "doc-g", "api-global.txt", visibility="global")

    response = await client.get("/documents/api/list")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    ids = {row["id"] for row in payload}
    assert ids == {"doc-a", "doc-g"}
    row = next(item for item in payload if item["id"] == "doc-a")
    assert row["name"] == "api-doc.txt"
    assert row["visibility"] == "user"
    assert row["ext"] == "txt"
    assert row["size"] == len("api-doc.txt") * 10
    assert row["chunks"] == 2
    assert row["created_at"]
