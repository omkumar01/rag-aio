"""Vanilla OpenAI adapter (api.openai.com)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .http import ApiKeyProvider
from .openai_compatible import OpenAICompatibleProvider


def _env_provider(name: str) -> Callable[[], str | None]:
    def _get() -> str | None:
        return os.environ.get(name)

    return _get


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI chat completions (https://api.openai.com/v1).

    ``api_key_provider`` defaults to the ``OPENAI_API_KEY`` environment variable
    (secrets by reference). Organization/project headers are passable via
    ``extra_headers`` / ``organization``.
    """

    def __init__(
        self,
        api_key_provider: ApiKeyProvider | None = None,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        transport: Any = None,
        supports_json_schema: bool = False,
        extra_headers: dict[str, str] | None = None,
        organization: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key_provider=api_key_provider or _env_provider("OPENAI_API_KEY"),
            timeout_s=timeout_s,
            transport=transport,
            supports_json_schema=supports_json_schema,
        )
        self.extra_headers: dict[str, str] = dict(extra_headers or {})
        if organization:
            self.extra_headers["OpenAI-Organization"] = organization

    def request_headers(self) -> dict[str, str]:
        headers = super().request_headers()
        headers.update(self.extra_headers)
        return headers
