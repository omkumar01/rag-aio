"""Test harness for the web-example app.

Adds ``web-example/src`` to ``sys.path`` (the example is not an installed
workspace member) and builds a per-test app over a throwaway data directory.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402  (must follow the sys.path bootstrap above)
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from web_example.db import init_db, sessionmaker_from  # noqa: E402
from web_example.main import create_app  # noqa: E402
from web_example.settings import AppSettings  # noqa: E402


@pytest.fixture
async def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path / "data", session_secret="test-secret")


@pytest.fixture
async def app(settings: AppSettings) -> AsyncIterator[Any]:
    application = create_app(settings)
    await init_db(application)
    yield application


@pytest.fixture
async def session(app: Any) -> AsyncIterator[AsyncSession]:
    maker = sessionmaker_from(app.state.engine)
    async with maker() as db_session:
        yield db_session


@pytest.fixture
async def client(app: Any) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.fixture
def register(client: AsyncClient):
    """Register + log in a user through the real HTTP flow."""

    async def _register(username: str, password: str = "hunter2secret") -> None:
        response = await client.post(
            "/signup", data={"username": username, "password": password}, follow_redirects=True
        )
        assert response.status_code == 200, response.text

    return _register
