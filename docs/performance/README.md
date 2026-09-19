# Performance

## Methodology

Benchmarks live in `benchmarks/` and measure: document throughput, OCR throughput,
chunking throughput, embedding throughput, indexing throughput, dense/sparse/hybrid
retrieval latency, reranking latency, context building latency, generation latency, end-
to-end query latency, RAM/VRAM usage, concurrency scaling, and cache hit performance.
Latency distributions report P50/P90/P95/P99 plus mean and median; cold-start and warm-
start are measured separately.

Results are published here per release with hardware, dataset, and configuration hash
recorded for reproducibility. Results will appear as modules land.
