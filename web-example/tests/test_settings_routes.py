"""Route-level tests for the settings pages (global defaults + per-user overrides)."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.services import settings_store


async def _user_id(sessionmaker: async_sessionmaker, username: str) -> str:
    """Resolve a user id by username (bound-parameter query only)."""
    async with sessionmaker() as session:
        rows = (
            await session.execute(
                text("SELECT id FROM users WHERE username = :name"), {"name": username}
            )
        ).fetchall()
    assert rows, f"user {username!r} not found"
    return str(rows[0][0])


async def _maker(client: AsyncClient) -> async_sessionmaker:
    return client._transport.app.state.sessionmaker


async def test_settings_requires_login(client: AsyncClient) -> None:
    response = await client.get("/settings", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


async def test_settings_page_renders_for_logged_in_user(client: AsyncClient, register) -> None:
    await register("alice")
    response = await client.get("/settings")
    assert response.status_code == 200
    assert "Global defaults" in response.text
    assert "Your overrides" in response.text


async def test_settings_page_read_only_global_for_non_admin(client: AsyncClient, register) -> None:
    await register("boss")  # first user is admin
    await client.post("/logout")
    await register("worker")
    response = await client.get("/settings")
    assert response.status_code == 200
    assert "read-only" in response.text
    assert 'action="/settings/global"' not in response.text


async def test_non_admin_gets_403_on_global_save(client: AsyncClient, register) -> None:
    await register("boss")
    await client.post("/logout")
    await register("worker")
    response = await client.post("/settings/global", data={"provider": "openai"})
    assert response.status_code == 403


async def test_admin_saves_global_defaults(
    client: AsyncClient, register, session: AsyncSession
) -> None:
    await register("boss")
    response = await client.post(
        "/settings/global",
        data={
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-test",
            "api_key_ref": "OPENAI_API_KEY",
            "temperature": "0.4",
            "retrieval_mode": "dense",
            "top_k": "5",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    config = await settings_store.get_global_config(session)
    assert config.generation.provider == "openai"
    assert config.generation.model == "gpt-test"
    assert config.generation.temperature == 0.4
    assert config.pipeline.retrieval.policy == "dense"
    # Secrets are stored by reference (env-var name) only, never a value.
    stored: dict[str, Any] = await settings_store.get_global_data(session)
    assert stored["generation"]["api_key_ref"] == "OPENAI_API_KEY"


async def test_user_saves_own_overrides(
    client: AsyncClient, register, session: AsyncSession
) -> None:
    await register("boss")
    await client.post("/settings/global", data={"temperature": "0.2"})
    await client.post("/logout")
    await register("worker")
    response = await client.post(
        "/settings/user",
        data={"temperature": "0.8", "provider": "anthropic"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    user_id = await _user_id(await _maker(client), "worker")
    overrides = await settings_store.get_user_overrides(session, user_id)
    assert overrides["generation"]["temperature"] == 0.8
    assert overrides["generation"]["provider"] == "anthropic"
    merged = await settings_store.get_merged_config(session, user_id)
    assert merged.generation.temperature == 0.8


async def test_invalid_override_returns_200_with_error(client: AsyncClient, register) -> None:
    await register("trixie")
    response = await client.post("/settings/user", data={"temperature": "very-hot"})
    assert response.status_code == 200
    assert "temperature" in response.text.lower()


async def test_invalid_provider_kind_rejected(client: AsyncClient, register) -> None:
    await register("trixie")
    response = await client.post("/settings/user", data={"provider": "not-a-provider"})
    assert response.status_code == 200
    assert "provider" in response.text.lower()


async def test_invalid_global_rejected_for_admin(client: AsyncClient, register) -> None:
    await register("boss")
    response = await client.post("/settings/global", data={"db_url": "mysql://nope"})
    assert response.status_code == 200
    assert "db_url" in response.text.lower()


async def test_clear_user_overrides(client: AsyncClient, register, session: AsyncSession) -> None:
    await register("wipe")
    await client.post("/settings/user", data={"temperature": "0.9"})
    response = await client.post("/settings/user/clear", follow_redirects=False)
    assert response.status_code == 303
    user_id = await _user_id(await _maker(client), "wipe")
    assert await settings_store.get_user_overrides(session, user_id) == {}


async def test_htmx_save_returns_partial(client: AsyncClient, register) -> None:
    await register("hxuser")
    response = await client.post(
        "/settings/user", data={"temperature": "0.5"}, headers={"HX-Request": "true"}
    )
    assert response.status_code == 200
    assert "settings-user" in response.text
    assert "0.5" in response.text


async def test_embedder_remote_warnings_rendered(client: AsyncClient, register) -> None:
    """The settings page ships the live remote-embedding requirement hints."""
    await register("warnuser")
    response = await client.get("/settings")
    assert response.status_code == 200
    assert "js-embedder-backend" in response.text
    assert "js-remote-requirements" in response.text
    assert "js-embedder-dim" in response.text
    assert "js-embedder-base-url" in response.text
    assert "/static/js/settings.js" in response.text


async def test_error_renders_above_save_button(client: AsyncClient, register) -> None:
    """Validation errors are positioned next to the save button, not the page top."""
    await register("erruser")
    response = await client.post(
        "/settings/global",
        data={"temperature": "not-a-number"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    error_at = response.text.index("form-error")
    button_at = response.text.index("Save global defaults")
    assert error_at < button_at
    # The error sits below the field sections (e.g. the generation fields)…
    assert error_at > response.text.index('name="provider"')
    # …and directly above the aggregated-requirements placeholder.
    assert 0 < response.text.index("js-remote-requirements") - error_at < 200
