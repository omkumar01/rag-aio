"""Chunker factory and registry."""

from __future__ import annotations

from collections.abc import Callable

from rag_core.protocols import Chunker, Tokenizer

from .base import BaseChunker, ChunkerConfig
from .fixed import FixedChunker
from .parent_child import ParentChildChunker
from .recursive import RecursiveChunker
from .sentence import SentenceChunker
from .structural import StructuralChunker
from .token import TokenChunker

__all__ = ["CHUNKERS", "create_chunker"]

# Maps strategy name -> chunker class.  Extensible: third-party chunkers
# can register by adding entries here.
CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "sentence": SentenceChunker,
    "token": TokenChunker,
    "recursive": RecursiveChunker,
    "structural": StructuralChunker,
    "parent_child": ParentChildChunker,
}

# Callable signature for custom chunker factories.
ChunkerFactory = Callable[[Tokenizer, ChunkerConfig], Chunker]


def create_chunker(config: ChunkerConfig, tokenizer: Tokenizer) -> Chunker:
    """Instantiate the chunker selected by *config.strategy*.

    Raises
    ------
    ValueError
        If the strategy is not registered.
    """
    cls = CHUNKERS.get(config.strategy)
    if cls is None:
        raise ValueError(
            f"unknown chunker strategy {config.strategy!r}; choices: {', '.join(sorted(CHUNKERS))}"
        )
    return cls(tokenizer, config)
