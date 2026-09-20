"""Deduplication of :class:`~rag_core.retrieval.RetrievalHit` candidates.

Two passes run when ``merge_overlapping`` is set:

1. **Exact dedup** by normalized-text hash (lower-cased, whitespace-collapsed).
   Hits without text fall back to their ``chunk_id``. The higher-scored hit in a
   group is kept; per-strategy scores from dropped members accumulate via a
   max-merge into ``strategy_scores``.
2. **Near-duplicate merging** via token-set Jaccard similarity over the hit
   text. Clusters whose members share enough tokens collapse to the
   highest-scored representative, again accumulating ``strategy_scores``.
"""

from __future__ import annotations

from rag_core.retrieval import RetrievalHit

__all__ = ["dedupe_hits"]


def _norm_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def _token_set(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(text.lower().split())


def _accumulate(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        if key not in dst or value > dst[key]:
            dst[key] = value


def _representative(group: list[RetrievalHit]) -> tuple[RetrievalHit, dict[str, float]]:
    """Pick the highest-scored hit and merge strategy_scores from the group."""
    best = max(group, key=lambda h: h.score)
    scores: dict[str, float] = {}
    for hit in group:
        _accumulate(scores, hit.strategy_scores)
    return best, scores


def _cluster(reps: list[RetrievalHit], similarity_threshold: float) -> list[list[int]]:
    """Group representative indices by Jaccard similarity of token sets."""
    n = len(reps)
    if n <= 1:
        return [[i] for i in range(n)] if n == 1 else []
    parent: dict[int, int] = {i: i for i in range(n)}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    token_sets = [_token_set(r.text) for r in reps]
    for i in range(n):
        for j in range(i + 1, n):
            union_set = token_sets[i] | token_sets[j]
            if not union_set:
                continue
            jaccard = len(token_sets[i] & token_sets[j]) / len(union_set)
            if jaccard >= similarity_threshold:
                union(i, j)
    clusters: dict[int, list[int]] = {}
    order: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
            order.append(root)
        clusters[root].append(i)
    return [clusters[root] for root in order]


def dedupe_hits(
    hits: list[RetrievalHit],
    merge_overlapping: bool = False,
    similarity_threshold: float = 0.9,
) -> list[RetrievalHit]:
    """Deduplicate retrieval hits, preserving order of first appearance.

    Args:
        hits: Candidate hits in priority order.
        merge_overlapping: When True, near-duplicates (Jaccard on token sets)
            above ``similarity_threshold`` are merged.
        similarity_threshold: Jaccard threshold for near-duplicate merging.

    Returns:
        Deduplicated hits. Each kept hit accumulates ``strategy_scores`` from
        dropped duplicates (max per strategy).
    """
    # 1. Exact dedup by normalized text hash (fall back to chunk_id).
    groups: dict[str, list[RetrievalHit]] = {}
    order_keys: list[str] = []
    for hit in hits:
        key = _norm_text(hit.text) if hit.text else hit.chunk_id
        key = key or hit.chunk_id
        if key not in groups:
            groups[key] = []
            order_keys.append(key)
        groups[key].append(hit)

    reps: list[RetrievalHit] = []
    for key in order_keys:
        best, scores = _representative(groups[key])
        if scores:
            best = best.model_copy(
                update={"strategy_scores": scores, "normalized_score": best.normalized_score}
            )
        reps.append(best)

    if not merge_overlapping:
        return reps

    # 2. Near-duplicate clustering via token-set Jaccard.
    clusters = _cluster(reps, similarity_threshold)
    result: list[RetrievalHit] = []
    for cluster in clusters:
        members = [reps[i] for i in cluster]
        best = max(members, key=lambda h: h.score)
        merged_scores: dict[str, float] = {}
        for member in members:
            _accumulate(merged_scores, member.strategy_scores)
        best = best.model_copy(update={"strategy_scores": merged_scores}) if merged_scores else best
        result.append(best)
    return result
