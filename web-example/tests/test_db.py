"""Unit tests for the web-example data layer."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import Conversation, Document, Message, SettingRow, UploadJob, User


async def test_tables_roundtrip(session: AsyncSession) -> None:
    user = User(id="u-1", username="alice", password_hash="h", is_admin=True)
    session.add(user)
    await session.flush()

    session.add(
        Document(
            id="doc-1",
            owner_id=user.id,
            visibility="user",
            name="notes.md",
            source_uri="file://notes.md",
            ext=".md",
            size_bytes=10,
            num_chunks=2,
            content_hash="abc",
        )
    )
    session.add(Conversation(id="c-1", user_id=user.id, title="chat 1"))
    await session.flush()

    session.add(
        Message(
            id="m-1",
            conversation_id="c-1",
            role="user",
            content="hello",
            attachments=[{"name": "notes.md"}],
        )
    )
    session.add(UploadJob(id="j-1", user_id=user.id, target="user", status="queued", total=1))
    await session.commit()

    found = await session.get(Document, "doc-1")
    assert found is not None
    assert found.visibility == "user"
    assert found.owner_id == user.id

    msg = await session.get(Message, "m-1")
    assert msg is not None
    attachments: list[dict[str, Any]] = msg.attachments or []
    assert attachments[0]["name"] == "notes.md"

    job = await session.get(UploadJob, "j-1")
    assert job is not None
    assert job.status == "queued"
    assert job.processed == 0

    convo = await session.get(Conversation, "c-1")
    assert convo is not None
    assert convo.title == "chat 1"


async def test_settings_json_roundtrip(session: AsyncSession) -> None:
    session.add(SettingRow(key="global", data={"generation": {"temperature": 0.5}}))
    await session.commit()

    row = await session.get(SettingRow, "global")
    assert row is not None
    data: dict[str, Any] = row.data
    assert data["generation"]["temperature"] == 0.5
