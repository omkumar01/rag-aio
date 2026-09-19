# rag-query

Query-intelligence and retrieval-execution layer: normalization, classification, rewriting,
expansion, multi-query, HyDE, decomposition, metadata extraction, and routing — with a
low-latency fast path that sends the original query straight to dense/sparse retrieval.
Independent strategies run concurrently via structured asyncio; every strategy is
measurable and disableable; no LLM rewrite is forced on any request.
