"""Unit tests for auth: signup, login, logout, route gating."""

from __future__ import annotations

from httpx import AsyncClient


async def test_landing_is_public(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "rag-aio" in response.text.lower()


async def test_protected_route_redirects_anonymous(client: AsyncClient) -> None:
    response = await client.get("/documents", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


async def test_htmx_request_gets_hx_redirect(client: AsyncClient) -> None:
    response = await client.get("/documents", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["HX-Redirect"] == "/login"


async def test_signup_first_user_is_admin_and_logged_in(client: AsyncClient, register) -> None:
    await register("alice")

    whoami = await client.get("/whoami")
    assert whoami.status_code == 200
    payload = whoami.json()
    assert payload["username"] == "alice"
    assert payload["is_admin"] is True


async def test_signup_duplicate_username_rejected(client: AsyncClient, register) -> None:
    await register("bob")
    await client.post("/logout")

    response = await client.post(
        "/signup", data={"username": "bob", "password": "another-pass-1"}, follow_redirects=True
    )
    assert response.status_code == 200
    assert "already" in response.text.lower()


async def test_login_success_and_failure(client: AsyncClient, register) -> None:
    await register("carol")
    await client.post("/logout")

    bad = await client.post(
        "/login", data={"username": "carol", "password": "wrong-password"}, follow_redirects=True
    )
    assert bad.status_code == 200
    assert "invalid" in bad.text.lower()

    good = await client.post(
        "/login", data={"username": "carol", "password": "hunter2secret"}, follow_redirects=False
    )
    assert good.status_code == 303
    assert good.headers["location"] == "/chat"


async def test_login_redirects_authenticated_user(client: AsyncClient, register) -> None:
    await register("dave")
    response = await client.get("/login", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/chat"


async def test_logout_clears_session(client: AsyncClient, register) -> None:
    await register("erin")
    await client.post("/logout", follow_redirects=False)
    response = await client.get("/whoami")
    assert response.status_code == 303
