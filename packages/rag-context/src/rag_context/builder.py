"""Token-budgeted context assembly.

:class:`ContextBuilderImpl` wires the pipeline together: score filtering,
deduplication, optional neighbor expansion, strategy-based ordering, greedy
token-budget selection, per-document caps, overlap merging, and citation
assignment, producing a :class:`~rag_core.context.Context` ready for rendering.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from rag_core.context import Context, ContextItem
from rag_core.protocols import ContextBuilder
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit

from .config import ContextConfig
from .dedup import dedupe_hits
from .expansion import NeighborProvider, expand_with_neighbors
from .ordering import order_by_strategy
from .tokenizer import ContextTokenizer

__all__ = ["ContextBuilderImpl", "sanitize_item_text"]

_NEWLINE_RUN = re.compile(r"\n{3,}")


def sanitize_item_text(text: str | None, max_chars: int) -> str:
    """Collapse whitespace, strip control characters, and cap length.

    * ``\\r`` normalized to ``\\n``;
    * control characters removed (newline and tab preserved);
    * runs of three or more newlines collapsed to two;
    * leading/trailing whitespace stripped;
    * length capped to ``max_chars`` characters.
    """
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "".join(
        ch for ch in cleaned if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C"
    )
    cleaned = _NEWLINE_RUN.sub("\n\n", cleaned)
    cleaned = cleaned.strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def _page_numbers(metadata: dict[str, Any]) -> list[int]:
    value = metadata.get("page_numbers")
    if isinstance(value, list):
        return [int(x) for x in value]
    return []


def _section_path(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("section_path")
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def _document_title(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("document_title") or metadata.get("title")
    if isinstance(value, str) and value:
        return value
    return None


def _source_uri(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("source_uri")
    if isinstance(value, str) and value:
        return value
    return None


def _hit_to_item(
    hit: RetrievalHit,
    config: ContextConfig,
    tokenizer: ContextTokenizer,
) -> ContextItem:
    metadata = hit.metadata or {}
    text = sanitize_item_text(hit.text, config.max_item_chars)
    return ContextItem(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        text=text,
        token_count=tokenizer.count(text),
        score=hit.score,
        page_numbers=_page_numbers(metadata),
        section_path=_section_path(metadata),
        document_title=_document_title(metadata),
        source_uri=_source_uri(metadata),
    )


def _greedy_select(
    items: list[ContextItem],
    effective_budget: int,
) -> tuple[list[ContextItem], bool]:
    """Select items in order, skipping those that no longer fit the budget."""
    selected: list[ContextItem] = []
    running = 0
    dropped = False
    for item in items:
        if item.token_count + running <= effective_budget:
            selected.append(item)
            running += item.token_count
        else:
            dropped = True
    return selected, dropped


def _cap_per_document(
    items: list[ContextItem],
    cap: int,
) -> tuple[list[ContextItem], bool]:
    counts: dict[str, int] = {}
    capped: list[ContextItem] = []
    dropped = False
    for item in items:
        seen = counts.get(item.document_id, 0)
        if seen >= cap:
            dropped = True
            continue
        counts[item.document_id] = seen + 1
        capped.append(item)
    return capped, dropped


def _overlap_ratio(a: str, b: str) -> float:
    """Longest suffix-of-``a``-equal-to-prefix-of-``b`` over the shorter length."""
    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return 0.0
    limit = min(len(a_tokens), len(b_tokens))
    for k in range(limit, 0, -1):
        if a_tokens[-k:] == b_tokens[:k]:
            return k / limit
    return 0.0


def _merge_overlapping(items: list[ContextItem], tokenizer: ContextTokenizer) -> list[ContextItem]:
    """Merge consecutive items whose trailing/leading text overlaps > 50%."""
    if not items:
        return []
    merged: list[ContextItem] = [items[0]]
    for item in items[1:]:
        current = merged[-1]
        ratio = _overlap_ratio(current.text, item.text)
        if ratio > 0.5:
            a_tokens = current.text.split()
            b_tokens = item.text.split()
            limit = min(len(a_tokens), len(b_tokens))
            k = 0
            for trial in range(limit, 0, -1):
                if a_tokens[-trial:] == b_tokens[:trial]:
                    k = trial
                    break
            union = " ".join(a_tokens + b_tokens[k:])
            current.text = union
            current.token_count = tokenizer.count(union)
            current.page_numbers = sorted(set(current.page_numbers) | set(item.page_numbers))
            current.section_path = list(
                dict.fromkeys(list(current.section_path) + list(item.section_path))
            )
        else:
            merged.append(item)
    return merged


class ContextBuilderImpl(ContextBuilder):
    """Concrete :class:`~rag_core.protocols.ContextBuilder` implementation."""

    def __init__(
        self,
        tokenizer: ContextTokenizer,
        config: ContextConfig,
        neighbor_provider: NeighborProvider | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._config = config
        self._neighbor_provider = neighbor_provider

    async def build(
        self,
        query: Query,
        hits: Sequence[RetrievalHit],
        token_budget: int,
    ) -> Context:
        config = self._config
        effective_budget = max(0, token_budget - config.reserve_for_answer)

        # 1. Score threshold filter.
        candidates: list[RetrievalHit] = list(hits)
        if config.min_score is not None:
            candidates = [hit for hit in candidates if hit.score >= config.min_score]

        # 2. Deduplication.
        if config.dedup:
            candidates = dedupe_hits(candidates, config.merge_overlapping)

        # 3. Neighbor expansion.
        if config.expand_neighbors and self._neighbor_provider is not None:
            candidates = expand_with_neighbors(
                candidates,
                self._neighbor_provider,
                config.neighbor_window,
            )

        # 4. Convert hits to context items.
        items = [_hit_to_item(hit, config, self._tokenizer) for hit in candidates]

        # 5. Order per strategy.
        ordered = order_by_strategy(items, config.strategy)

        # 6. Greedy token-budget selection.
        selected, budget_dropped = _greedy_select(ordered, effective_budget)

        # 7. Per-document cap.
        if config.max_per_document > 0:
            selected, cap_dropped = _cap_per_document(selected, config.max_per_document)
        else:
            cap_dropped = False

        # 8. Overlap merging.
        merged = _merge_overlapping(selected, self._tokenizer)

        # 9. Citation assignment.
        if config.citation_style == "numeric":
            for index, item in enumerate(merged, start=1):
                item.citation_id = str(index)
        else:
            for item in merged:
                item.citation_id = None

        truncated = budget_dropped or cap_dropped
        return Context(
            items=merged,
            token_budget=token_budget,
            strategy=config.strategy,
            truncated=truncated,
        )
