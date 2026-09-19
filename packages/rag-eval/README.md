# rag-eval

Evaluation framework measuring retrieval quality (Recall@K, Precision@K, Hit Rate, MRR,
nDCG, Average Precision), reranking quality, context quality, answer quality
(faithfulness/groundedness/relevance/citation correctness), and latency/resource usage.
JSONL/CSV/Parquet datasets, golden IDs/answers, reproducible experiment configs persisted
with model versions, per-stage metrics, and explicit separation of retrieval failure from
generation failure.
