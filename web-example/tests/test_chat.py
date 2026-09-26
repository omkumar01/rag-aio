"""Route-level tests for the chat UI endpoints (SSE streaming, ownership, limits)."""

from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse an SSE body into a flat list of JSON event payloads."""
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


async def _ask(client: AsyncClient, **kwargs: Any) -> list[dict[str, Any]]:
    response = await client.post("/chat/ask", **kwargs)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


async def test_chat_page_redirects_anonymous(client: AsyncClient) -> None:
    response = await client.get("/chat", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


async def test_conversations_partial_redirects_anonymous(client: AsyncClient) -> None:
    response = await client.get("/chat/conversations", follow_redirects=False)
    assert response.status_code == 303


async def test_ask_streams_meta_delta_done_and_persists(client: AsyncClient, register) -> None:
    await register("alice")
    events = await _ask(
        client,
        data={"message": "Where does the zebra fox nap?"},
        files={
            "doc_files": (
                "notes.txt",
                b"The zebra fox naps in the sunny loft every afternoon.",
                "text/plain",
            )
        },
    )
    types = [event["type"] for event in events]
    assert "meta" in types
    assert "delta" in types
    assert types[-1] == "done"

    meta = next(event for event in events if event["type"] == "meta")
    conversation_id = str(meta["conversation_id"])
    assert meta["title"] == "Where does the zebra fox nap?"

    answer = "".join(str(event.get("text", "")) for event in events if event["type"] == "delta")
    assert "zebra" in answer

    sidebar = await client.get("/chat/conversations")
    assert sidebar.status_code == 200
    assert "Where does the zebra fox nap?" in sidebar.text

    thread = await client.get(f"/chat/conversations/{conversation_id}/messages")
    assert thread.status_code == 200
    assert "zebra" in thread.text
    assert "notes.txt" in thread.text  # attachment chip on the user message


async def test_ask_without_attachment_persists_conversation(client: AsyncClient, register) -> None:
    await register("alice")
    events = await _ask(client, data={"message": "hello there, who are you?"})
    assert [event["type"] for event in events][-1] == "done"
    sidebar = await client.get("/chat/conversations")
    assert "hello there, who are you?" in sidebar.text


async def test_user_b_cannot_read_user_a_conversation(client: AsyncClient, register) -> None:
    await register("alice")
    events = await _ask(client, data={"message": "alice private question"})
    meta = next(event for event in events if event["type"] == "meta")
    conversation_id = str(meta["conversation_id"])

    await client.post("/logout")
    await register("bob")
    response = await client.get(f"/chat/conversations/{conversation_id}/messages")
    assert response.status_code in (303, 404)


async def test_more_than_five_doc_attachments_rejected(client: AsyncClient, register) -> None:
    await register("alice")
    files = [("doc_files", (f"file{i}.txt", b"tiny content", "text/plain")) for i in range(6)]
    response = await client.post("/chat/ask", data={"message": "hi"}, files=files)
    assert response.status_code == 413


async def test_more_than_five_image_attachments_rejected(client: AsyncClient, register) -> None:
    await register("alice")
    files = [("image_files", (f"img{i}.png", b"\x89PNG", "image/png")) for i in range(6)]
    response = await client.post("/chat/ask", data={"message": "hi"}, files=files)
    assert response.status_code == 413


async def test_five_attachments_of_each_type_accepted(client: AsyncClient, register) -> None:
    await register("alice")
    files = [("doc_files", (f"file{i}.txt", b"content for file", "text/plain")) for i in range(5)]
    files += [("image_files", (f"img{i}.png", b"\x89PNG", "image/png")) for i in range(5)]
    events = await _ask(client, data={"message": "summarize the uploads"}, files=files)
    assert [event["type"] for event in events][-1] == "done"


async def test_delete_conversation_removes_messages(client: AsyncClient, register) -> None:
    await register("alice")
    events = await _ask(client, data={"message": "to be deleted"})
    meta = next(event for event in events if event["type"] == "meta")
    conversation_id = str(meta["conversation_id"])

    response = await client.delete(f"/chat/conversations/{conversation_id}")
    assert response.status_code == 200

    thread = await client.get(f"/chat/conversations/{conversation_id}/messages")
    assert thread.status_code == 404

    sidebar = await client.get("/chat/conversations")
    assert "to be deleted" not in sidebar.text


async def test_ask_without_rag_engine_emits_error_event(
    client: AsyncClient, register, app: Any
) -> None:
    await register("alice")
    original = app.state.rag
    app.state.rag = None
    try:
        response = await client.post("/chat/ask", data={"message": "hi"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert any(event.get("type") == "error" for event in events)
    finally:
        app.state.rag = original
