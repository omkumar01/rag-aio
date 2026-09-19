# rag-mass-inject

High-throughput bulk ingestion: directories, file lists, sitemaps/URLs, object-storage
locations. Streaming bounded batches (never unbounded RAM), per-stage concurrency (read,
parse, OCR, chunk, embed, index), deduplication, incremental ingestion, job queues with
checkpoints and resumability, retry policies, dead-letter handling, progress reporting,
and cancellation.
