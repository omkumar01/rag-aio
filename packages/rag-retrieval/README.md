# rag-retrieval

Dedicated retrieval engine for rag-aio: dense vector retrieval, sparse/BM25 retrieval
(bm25s), hybrid retrieval with Reciprocal Rank Fusion and weighted fusion, metadata/
namespace/date/access-control filtering, score thresholds, top-K control, parent/child
expansion, and full retrieval explainability (per-hit source strategy, raw and normalized
scores, rank, filter decisions, model provenance). Backends: Qdrant (via rag-db-handler),
FAISS (extra).
