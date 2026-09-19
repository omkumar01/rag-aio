"""Secrets-safety: API keys must never leak into error messages/details."""

from __future__ import annotations

import httpx
import pytest
from rag_core.errors import ProviderError
from rag_generation import OpenAICompatibleProvider


async def test_api_key_not_in_exception(make_request, wire) -> None:
    captured: dict[str, object] = {}
    # built at runtime so this fixture is not a literal credential
    fake_key = "sk-" + "secret-12345"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    prov = wire(OpenAICompatibleProvider, handler, api_key=fake_key)
    with pytest.raises(ProviderError) as exc:
        await prov.complete(make_request())

    blob = str(exc.value) + repr(exc.value.details)
    assert fake_key not in blob
    assert exc.value.code == "auth"
    # The header is still sent over the wire, just never surfaced in errors.
    assert captured["headers"]["authorization"] == f"Bearer {fake_key}"
