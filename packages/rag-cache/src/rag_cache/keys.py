"""Cache key construction.

Keys are deterministic, namespaced, and embed a content/config hash so that
semantically equivalent inputs map to the same cache entry while different
inputs (different namespace, different parts) never collide.
"""

from __future__ import annotations

from rag_core.ids import config_hash


def cache_key(namespace: str, **parts: object) -> str:
    """Build a deterministic cache key: ``namespace:<sha256-of-canonical-parts>``.

    The digest is computed with :func:`rag_core.ids.config_hash`, which
    canonicalizes ``parts`` (``json.dumps`` with ``sort_keys=True`` and
    ``default=str``). That makes the key:

    * **deterministic** -- identical inputs always produce the same key,
    * **order-insensitive** for dict/parameter parts (key order does not matter),
    * **namespace-scoped** -- the ``namespace`` is a literal prefix and is therefore
      *not* folded into the hash; two distinct namespaces never collide even when
      their ``parts`` are identical.

    Args:
        namespace: Logical grouping / prefix (e.g. a stage name or config hash).
        **parts: Arbitrary values that identify the cached artifact. Dicts,
            Pydantic models, JSON scalars, etc. are all accepted because
            ``config_hash`` falls back to ``str`` for non-JSON types.

    Returns:
        A string of the form ``namespace:<64 hex chars>``.
    """
    digest = config_hash(parts)
    return f"{namespace}:{digest}"
