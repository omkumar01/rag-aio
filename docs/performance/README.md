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
recorded for reproducibility. See [benchmarks/README.md](../../benchmarks/README.md) for
the full harness methodology (cold/warm split, percentile definition, workload
determinism) and for how to reproduce: `uv run pytest benchmarks -m benchmark`.

## Measured results

Wave-8 run — 2026-09-24, dev workstation (Windows 11, 12 CPUs, Python 3.14 interpreter),
mock/offline scenarios plus local LM Studio; config hash `6183dc5d35a518bb`
(source: [final-report.md](../development/final-report.md) §4):

| Scenario | Workload | Result |
| --- | --- | ---: |
| doc parse + chunk | 30 synthetic docs | 114 ms |
| mock embedding | 200 chunks | 82 ms |
| FastEmbed (`bge-small-en-v1.5`) | 32 texts (model cached) | 513 ms |
| dense retrieval | 200 mock-embedded chunks, candidate_k=50 | 68.7 ms mean |
| hybrid retrieval (dense + BM25 + RRF) | 200 chunks | 64.7 ms mean |
| rerank (heuristic) | 50 candidates | 1.8 ms |
| context build | 50 hits, 2048-token budget | 5.9 ms |
| full mock query (`RAGConfig.mock()`) | ingest 10 docs + ask | 25.5 ms |
| generation | local LM Studio, small `max_tokens` | ~35 ms mean |

All 8 documented scenario families ran (generation auto-skips when `localhost:1234` is
unreachable; here it answered).

Caveats:

- Absolute numbers are machine-dependent — use relative comparisons and trend lines, not
  the raw figures. Numbers from different hardware or different `config_hash` values must
  not be compared directly (see `benchmarks/README.md`).
- Peak-RAM figures come from `tracemalloc` and cover Python-level allocations only —
  native buffers (ONNX runtime, model weights, Qdrant storage) are invisible, so they are
  a lower bound on real process memory. VRAM is not measured.
- The scaled multi-service topology (Qdrant server, PostgreSQL, Redis behind FastAPI
  services) is not exercised by the benchmark suite.
