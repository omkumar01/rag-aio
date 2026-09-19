# ADR-0002: Protocol-based contracts with adapter registries

**Status:** Accepted | **Date:** 2026-09-19

## Context
Modules must compose without coupling to vendors or to each other's concrete classes. Inheritance hierarchies are brittle across independently evolving modules.

## Decision
All cross-module capabilities are `typing.Protocol` definitions (runtime-checkable) owned by `rag-core`: DocumentLoader, DocumentParser, OCRProcessor, Chunker, Tokenizer, Embedder, SparseEmbedder, Indexer, VectorStore, DocumentStore, Cache, Retriever, HybridRetriever, FusionStrategy, Reranker, QueryStrategy, ContextBuilder, Generator, LLMProvider, EvaluationMetric, Observer. Implementations are adapters registered via explicit registries plus entry points. Composition over inheritance; Pydantic v2 models for boundary data; dataclasses/numpy for hot-path internals.

## Consequences
- Vendor types never appear in rag-core or in other modules' public APIs.
- Replacing FastEmbed/Qdrant/OpenAI with alternatives is a configuration + dependency-injection change.
- Contract tests assert protocol conformance for every adapter.
