"""OpenAI-compatible remote reranker with runtime endpoint auto-detection.

ADR-0004 targets LM Studio as the default local provider. LM Studio (and the
qwen3-reranker family) may expose either a native ``/rerank`` endpoint
(Jina/Cohere-shaped) *or* only ``/v1/chat/completions``. Rather than
guessing, :class:`RemoteReranker` probes ``POST {base_url}/rerank`` on its
first rerank call and permanently selects a scoring strategy:

* **rerank mode** - POST ``/rerank`` with ``{model, query, documents}``;
  parse ``results[i].score`` (index-aligned to handle re-ordered responses).
* **chat mode** - fall back to POST ``/chat/completions`` with a scoring
  template that asks the model to emit a JSON array of 0-10 relevance scores;
  the first ``[...]`` block is extracted and mapped to 0-1.

Detection runs once and is cached in ``self._mode``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import httpx
from rag_core import Query, RerankError, RetrievalHit

from .base import Scorer, rerank_candidates
from .config import RerankConfig

_RerankMode = Literal["rerank", "chat"]

__all__ = ["RemoteReranker"]


class RemoteReranker:
    """Reranker backed by an OpenAI-compatible HTTP provider.

    ``api_key_ref`` is the name of an environment variable holding the bearer
    token. The secret is resolved at request time and never stored on the
    instance, so it cannot leak via ``repr``/logs/serialization.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_ref: str | None = None,
        config: RerankConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_ref = api_key_ref
        self._config = config or RerankConfig()
        self._transport: httpx.AsyncBaseTransport | None = transport
        self._client: httpx.AsyncClient | None = None
        self._mode: _RerankMode | None = None

    async def rerank(
        self,
        query: Query,
        candidates: list[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if not candidates:
            return []
        await self._probe()
        scorer = self._make_scorer()
        cfg = self._config.model_copy(update={"top_k": top_k})
        hits = await rerank_candidates(query.text, list(candidates), scorer, cfg)
        for hit in hits:
            hit.model = self._model
        return hits

    async def aclose(self) -> None:
        """Close the underlying HTTP client (no-op if never opened)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def mode(self) -> _RerankMode | None:
        """Detected scoring mode (``"rerank"`` or ``"chat"``) once probed."""
        return self._mode

    def __repr__(self) -> str:
        return (
            f"RemoteReranker(base_url={self._base_url!r}, "
            f"model={self._model!r}, api_key_ref={self._api_key_ref!r})"
        )

    # -- internals -----------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_s),
                transport=self._transport,
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key_ref is not None:
            key = os.environ.get(self._api_key_ref)
            if key is None:
                raise RerankError(f"api_key_ref env var {self._api_key_ref!r} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def _probe(self) -> None:
        """Detect ``/rerank`` support; on 404/405/422 fall back to chat mode."""
        if self._mode is not None:
            return
        client = self._get_client()
        headers = self._auth_headers()
        body: dict[str, Any] = {
            "model": self._model,
            "query": "probe",
            "documents": ["probe"],
        }
        try:
            resp = await client.post(f"{self._base_url}/rerank", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise RerankError(f"rerank endpoint probe failed: {exc}") from exc
        if resp.status_code == 200:
            self._mode = "rerank"
        elif resp.status_code in (404, 405, 422):
            self._mode = "chat"
        else:
            raise RerankError(f"unexpected status {resp.status_code} from /rerank probe")

    def _make_scorer(self) -> Scorer:
        async def scorer(query: str, texts: list[str]) -> list[float]:
            if self._mode == "chat":
                return await self._chat_request(query, texts)
            return await self._rerank_request(query, texts)

        return scorer

    async def _rerank_request(self, query: str, texts: list[str]) -> list[float]:
        client = self._get_client()
        headers = self._auth_headers()
        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": texts,
        }
        resp = await client.post(f"{self._base_url}/rerank", json=body, headers=headers)
        if resp.status_code != 200:
            raise RerankError(f"/rerank failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        return self._parse_rerank(data, len(texts))

    @staticmethod
    def _parse_rerank(data: Any, n: int) -> list[float]:
        """Parse Jina/Cohere-style ``/rerank`` results into an index-aligned list."""
        results = data.get("results") or data.get("data") or []
        scores = [0.0] * n
        indexed = False
        for item in results:
            idx = item.get("index")
            score = item.get("score")
            if (
                isinstance(idx, int)
                and not isinstance(idx, bool)
                and isinstance(score, (int, float))
            ):
                scores[idx] = float(score)
                indexed = True
        if not indexed and len(results) == n:
            for i, item in enumerate(results):
                score = item.get("score")
                if isinstance(score, (int, float)):
                    scores[i] = float(score)
        return scores

    async def _chat_request(self, query: str, texts: list[str]) -> list[float]:
        client = self._get_client()
        headers = self._auth_headers()
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        prompt = (
            f"Query: {query}\n\n"
            "For each numbered document, output a relevance score from 0 to 10 "
            "as a JSON array of numbers in document order, e.g. [7.5, 3.0, 8.2]. "
            "Output ONLY the JSON array and nothing else.\n\n"
            f"Documents:\n{numbered}\n\nScores:"
        )
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        resp = await client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
        if resp.status_code != 200:
            raise RerankError(f"/chat/completions failed: HTTP {resp.status_code}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RerankError("malformed chat/completions response") from exc
        if not isinstance(content, str):
            raise RerankError("chat completion content is not a string")
        return self._parse_chat_scores(content, len(texts))

    @staticmethod
    def _parse_chat_scores(content: str, n: int) -> list[float]:
        """Extract a JSON float array (0-10) from chat text and map to 0-1."""
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise RerankError("could not locate JSON score array in chat response")
        raw = content[start : end + 1]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RerankError(f"invalid JSON score array: {raw[:120]!r}") from exc
        if not isinstance(parsed, list) or len(parsed) != n:
            got = len(parsed) if isinstance(parsed, list) else "non-list"
            raise RerankError(f"expected {n} scores, got {got}")
        scores: list[float] = []
        for value in parsed:
            try:
                scores.append(float(value) / 10.0)
            except (TypeError, ValueError) as exc:
                raise RerankError(f"invalid score value: {value!r}") from exc
        return scores
