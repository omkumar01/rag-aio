# rag-rerank

Second-stage relevance optimization for rag-aio, applied to small candidate sets (reranking
is more expensive than first-stage retrieval). Adapters: local CrossEncoder
(sentence-transformers extra) and OpenAI-compatible remote rerankers (e.g. qwen3-reranker
via LM Studio). Batching, score thresholds, top-N, score normalization, duplicate removal,
and preservation of both original retrieval scores and reranker scores for explainability.
