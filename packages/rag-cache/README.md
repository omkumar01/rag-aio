# rag-cache

Multi-level caching (memory, filesystem, sqlite; Redis behind the `redis` extra) for
parsing, OCR, chunking, embeddings, retrieval, reranking, and generation results. Cache
keys include model/config/content hashes so stale artifacts are never reused. Per-stage
opt-in, TTL/LRU/size limits, namespaces, statistics and hit/miss metrics.
