"""Rendering of assembled contexts into prompt-sized text blocks.

``render_context`` only formats the retrieved evidence selected by a
:class:`~rag_context.builder.ContextBuilderImpl`. It never assembles system
instructions or prompt templates — that composition is the caller's responsibility
so a context can be reused across different prompts.
"""

from __future__ import annotations

from rag_core.context import Context

__all__ = ["render_context"]


def render_context(context: Context, style: str = "numbered") -> str:
    """Render a context's items as ``[n] text`` blocks separated by blank lines.

    The leading ``[n]`` matches the ``citation_id`` assigned during assembly
    (numeric style). Blocks are emitted in the context's final item order.
    """
    if style != "numbered":
        # Only the numbered style is defined; silently fall back to it rather
        # than producing mismatched citation markers.
        style = "numbered"
    blocks: list[str] = []
    for index, item in enumerate(context.items, start=1):
        text = item.text if item.text else ""
        blocks.append(f"[{index}] {text}")
    return "\n\n".join(blocks)
