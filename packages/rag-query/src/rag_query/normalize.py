"""Query text normalization: unicode folding, whitespace and control-char cleanup."""

from __future__ import annotations

import re
import unicodedata

# Control characters in the C0 (0x00-0x1f) and C1 (0x7f-0x9f) ranges.
_CONTROL_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# Any run of whitespace.
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


def normalize_query(text: str, lowercase: bool = False) -> str:
    """Normalize a query string.

    Applies unicode NFKC folding, removes control characters, collapses
    internal whitespace runs to a single space and strips the result.

    Args:
        text: Raw query text.
        lowercase: When ``True`` lower-case the result.

    Returns:
        The normalized query string.
    """
    folded = unicodedata.normalize("NFKC", text)
    stripped = _CONTROL_RE.sub("", folded)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    if lowercase:
        collapsed = collapsed.lower()
    return collapsed
