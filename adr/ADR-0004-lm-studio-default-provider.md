# ADR-0004: OpenAI-compatible (LM Studio) as default local model provider

**Status:** Accepted | **Date:** 2026-09-19

## Context
The development workstation runs LM Studio at localhost:1234 exposing an OpenAI-compatible API with generation, embedding, reranking, and OCR/VLM models. Users may alternatively use OpenAI, Anthropic, Gemini, Ollama, vLLM, or llama.cpp servers.

## Decision
The OpenAI-compatible adapter is the reference implementation of the Generator/Embedder/Reranker/VLM contracts, defaulting to `http://localhost:1234/v1`. Dedicated adapters exist for OpenAI, Anthropic, Google/Gemini, and Ollama native APIs. Examples, e2e smoke tests, and default profiles target LM Studio; e2e tests skip automatically when the endpoint is unreachable, and CI uses mocks only.

## Consequences
- No cloud API keys are needed for local development or CI.
- Provider-specific features (reasoning controls, constrained JSON) are capability-discovered and degrade gracefully.
