"""Query engine: normalization, classification and concurrent strategy orchestration."""

from __future__ import annotations

import asyncio
import logging
import time

from rag_core.base import RagBaseModel
from rag_core.errors import ConfigError
from rag_core.protocols import QueryStrategy
from rag_core.queries import Query, QueryVariant

from .classify import classify_query
from .config import QueryConfig
from .normalize import normalize_query
from .strategies._base import LLM
from .strategies.decompose import DecomposeStrategy
from .strategies.expansion import ExpansionStrategy
from .strategies.hyde import HyDEStrategy
from .strategies.rewrite import RewriteStrategy

logger = logging.getLogger(__name__)


class QueryResult(RagBaseModel):
    """Outcome of running :meth:`QueryEngine.process`."""

    query: Query
    variants: list[QueryVariant]
    query_class: str
    timings_ms: dict[str, float]


class QueryEngine:
    """Orchestrates query normalization, classification and transformation.

    Strategies run concurrently via :class:`asyncio.TaskGroup`. The result
    always includes a ``kind="original"`` variant; remaining variants are
    de-duplicated (case-insensitive) and capped at ``QueryConfig.max_variants``.

    With a default (all flags off) configuration *no* LLM call is ever made and
    ``process`` returns solely the original variant.
    """

    def __init__(
        self,
        config: QueryConfig,
        strategies: list[QueryStrategy] | None = None,
        *,
        llm: LLM | None = None,
    ) -> None:
        self.config = config
        self._llm = llm
        if strategies is not None:
            self._strategies: list[QueryStrategy] = list(strategies)
        else:
            self._strategies = self._build_default_strategies()

    def _build_default_strategies(self) -> list[QueryStrategy]:
        """Construct strategies from config flags.

        ``deterministic_only`` forces every strategy to run without an LLM.
        HyDE has no deterministic form, so enabling it under that flag raises
        :class:`~rag_core.errors.ConfigError`.
        """
        llm = None if self.config.deterministic_only else self._llm
        strategies: list[QueryStrategy] = []
        if self.config.expand:
            strategies.append(ExpansionStrategy(llm=llm, count=self.config.expansion_count))
        if self.config.rewrite:
            strategies.append(RewriteStrategy(llm=llm))
        if self.config.decompose:
            strategies.append(DecomposeStrategy(llm=llm, max_subqueries=self.config.max_variants))
        if self.config.hyde:
            if self.config.deterministic_only:
                raise ConfigError(
                    "hyde strategy requires an LLM but deterministic_only is True",
                )
            if llm is None:
                raise ConfigError(
                    "hyde strategy requires an LLM but none was provided",
                )
            strategies.append(HyDEStrategy(llm=llm))
        return strategies

    async def process(self, query: Query) -> QueryResult:
        """Normalize, classify and transform *query*, returning all variants."""
        timings: dict[str, float] = {}
        started = time.perf_counter()

        text = normalize_query(query.text) if self.config.normalize else query.text
        timings["normalize_ms"] = (time.perf_counter() - started) * 1000.0

        classify_started = time.perf_counter()
        query_class = classify_query(text)
        timings["classify_ms"] = (time.perf_counter() - classify_started) * 1000.0

        strategy_variants: list[QueryVariant] = []
        run_strategies = self._strategies and self.config.min_query_length <= len(text.split())
        if run_strategies:
            transformed = await self._run_strategies(query, text)
            strategy_variants = list(transformed)

        variants = self._assemble(query, text, strategy_variants)

        timings["process_ms"] = (time.perf_counter() - started) * 1000.0
        result_query = query.model_copy(update={"text": text, "variants": variants})
        return QueryResult(
            query=result_query,
            variants=variants,
            query_class=query_class.value,
            timings_ms=timings,
        )

    async def _run_strategies(self, query: Query, text: str) -> list[QueryVariant]:
        """Run all strategies concurrently against a normalized query copy."""
        normalized = query.model_copy(update={"text": text, "variants": []})
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(strategy.transform(normalized))
                for strategy in self._strategies
            ]
        combined: list[QueryVariant] = []
        for task in tasks:
            combined.extend(task.result())
        return combined

    def _assemble(
        self,
        query: Query,
        text: str,
        strategy_variants: list[QueryVariant],
    ) -> list[QueryVariant]:
        """Build the final ordered, de-duplicated, capped variant list."""
        original = QueryVariant(
            query_id=query.id,
            text=text,
            kind="original",
            strategy="original",
        )
        combined: list[QueryVariant] = [original, *strategy_variants]

        seen: set[str] = set()
        deduped: list[QueryVariant] = []
        for variant in combined:
            key = variant.text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variant)

        return deduped[: self.config.max_variants]
