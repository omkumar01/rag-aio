"""Per-strategy ordering of :class:`~rag_core.context.ContextItem` lists.

Each strategy is a pure function ``order(items) -> list[ContextItem]``. Ordering is
deterministic and stable: ties keep the input order. The dispatcher
:func:`order_by_strategy` maps a strategy name (as found in
:class:`~rag_context.config.ContextConfig`) to its function.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_core.context import ContextItem

__all__ = [
    "STRATEGIES",
    "OrderFn",
    "chronological",
    "diversity",
    "document_grouped",
    "order_by_strategy",
    "relevance_first",
    "section_aware",
]

OrderFn = Callable[[list[ContextItem]], list[ContextItem]]


def relevance_first(items: list[ContextItem]) -> list[ContextItem]:
    """Order by score descending; items with a ``None`` score sort last (stable)."""

    def key(item: ContextItem) -> tuple[int, float]:
        if item.score is None:
            return (1, 0.0)
        return (0, -(item.score))

    return sorted(items, key=key)


def chronological(items: list[ContextItem]) -> list[ContextItem]:
    """Order by document id, then by the first page number (stable)."""

    def key(item: ContextItem) -> tuple[str, int]:
        page = item.page_numbers[0] if item.page_numbers else 0
        return (item.document_id, page)

    return sorted(items, key=key)


def section_aware(items: list[ContextItem]) -> list[ContextItem]:
    """Order by document id, then by section path (stable)."""

    def key(item: ContextItem) -> tuple[str, tuple[str, ...]]:
        path = tuple(item.section_path)
        return (item.document_id, path)

    return sorted(items, key=key)


def document_grouped(items: list[ContextItem]) -> list[ContextItem]:
    """Group by document; documents ordered by their best score (desc).

    Within a document the original input order is preserved. Documents with no
    scored items sort after scored ones.
    """
    by_doc: dict[str, list[ContextItem]] = {}
    for item in items:
        by_doc.setdefault(item.document_id, []).append(item)

    def best_score(doc_id: str) -> float:
        scored = [it.score for it in by_doc[doc_id] if it.score is not None]
        return max(scored) if scored else 0.0

    doc_ids = list(by_doc.keys())
    doc_ids.sort(key=lambda d: (-best_score(d), d))
    result: list[ContextItem] = []
    for doc_id in doc_ids:
        result.extend(by_doc[doc_id])
    return result


def diversity(items: list[ContextItem]) -> list[ContextItem]:
    """Round-robin across documents ordered by their best score (desc)."""
    by_doc: dict[str, list[ContextItem]] = {}
    for item in items:
        by_doc.setdefault(item.document_id, []).append(item)

    def best_score(doc_id: str) -> float:
        scored = [it.score for it in by_doc[doc_id] if it.score is not None]
        return max(scored) if scored else 0.0

    doc_ids = list(by_doc.keys())
    doc_ids.sort(key=lambda d: (-best_score(d), d))
    queues: list[list[ContextItem]] = [by_doc[d] for d in doc_ids]
    result: list[ContextItem] = []
    while True:
        progressed = False
        for queue in queues:
            if queue:
                result.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return result


STRATEGIES: dict[str, OrderFn] = {
    "relevance_first": relevance_first,
    "chronological": chronological,
    "section_aware": section_aware,
    "document_grouped": document_grouped,
    "diversity": diversity,
}


def order_by_strategy(items: list[ContextItem], strategy: str) -> list[ContextItem]:
    """Dispatch to the ordering function for ``strategy``."""
    fn = STRATEGIES.get(strategy)
    if fn is None:
        msg = f"unknown ordering strategy: {strategy!r}"
        raise ValueError(msg)
    return fn(list(items))
