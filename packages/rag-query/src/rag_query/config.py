"""Configuration for the query-intelligence engine."""

from __future__ import annotations

from rag_core.base import RagBaseModel


class QueryConfig(RagBaseModel):
    """Tunable flags controlling query transformation and strategy selection.

    ``deterministic_only`` gates every LLM-dependent strategy: when ``True``
    (the default) any strategy that *requires* an LLM is refused, so the engine
    never makes a network call. Individual strategies also expose deterministic
    fallbacks that are used automatically when no LLM is supplied.
    """

    normalize: bool = True
    expand: bool = False
    rewrite: bool = False
    hyde: bool = False
    decompose: bool = False
    max_variants: int = 3
    expansion_count: int = 2
    deterministic_only: bool = True
    min_query_length: int = 2
