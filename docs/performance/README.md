# Performance

## Methodology

Benchmarks live in `benchmarks/` and measure: document parse+chunk throughput,
embedding throughput (mock and FastEmbed), dense and hybrid retrieval latency,
reranking latency, context building latency, generation latency (local
OpenAI-compatible server, auto-skipped when unreachable), end-to-end query
latency, and RAM usage (Python allocations via `tracemalloc`). Latency
distributions report P50/P90/P95/P99 plus mean and median; cold-start and warm-
start are measured separately.

Planned, not yet measured: OCR throughput, standalone chunking and indexing
throughput, sparse/BM25-only retrieval latency, concurrency scaling, cache hit
performance, and VRAM usage.

Results are published here per release with hardware, dataset, and configuration hash
recorded for reproducibility. Results will appear as modules land.
