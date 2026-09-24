# rag-aio Benchmarks

Custom pytest-based benchmark harness for the rag-aio platform (Wave 8).
Deliberately **not** pytest-benchmark — the harness is pure standard library
plus pytest, so the suite stays dependency-light and CI-friendly.

## How to run

```bash
uv run pytest benchmarks -m benchmark   # full benchmark suite (primary entry point)
uv run pytest benchmarks -m unit        # harness math unit tests only (fast, CI-safe)
```

Scenario sources live in `test_scenarios.py`; the harness itself in
`harness.py`, with its percentile/timing unit tests in `test_harness.py`.

## What is measured

| benchmark | workload | stage |
| --- | --- | --- |
| `doc_throughput` | 30 synthetic text docs (~2-3 KB each) | parse (`rag-doc-handler` `default_pipeline`) + recursive chunking |
| `embedding_throughput_mock` | 200 chunks | mock dense + sparse embedding and in-memory indexing |
| `embedding_throughput_fastembed` | 32 chunks | `FastEmbedDense` (bge-small-en-v1.5); skipped unless the model is already cached locally |
| `retrieval_dense_latency` | 200 mock-embedded chunks | `DenseRetriever` (embed + cosine search, `candidate_k=50`) |
| `retrieval_hybrid_latency` | 200 chunks | `HybridRetriever` (dense + BM25) fused with `ReciprocalRankFusion`; cold run includes the BM25 index build |
| `rerank_latency` | 50 candidates | `RerankPipeline` + `HeuristicReranker` (Jaccard baseline) |
| `context_build_latency` | 50 hits | `ContextBuilderImpl` with `WhitespaceCounter` under a 2048-token budget |
| `full_query_mock` | 10 docs ingested, 1 query | end-to-end `RAG.from_config(RAGConfig.mock())` |
| `generation_latency` | tiny prompt, `max_tokens=8` | `OpenAICompatibleProvider` against local LM Studio at `localhost:1234/v1` — auto-skipped when the endpoint does not answer within 2 s |

## Methodology

- **Runs per scenario**: each measured run executes the scenario function once.
  Every scenario performs **one cold (first, unwarmed) run**, then `warmup`
  discarded runs (usually 1-2) so lazy initialization — parser registries,
  BM25 index builds, model loading — lands in the cold/warmup runs, then
  `iterations` timed runs (5-20 depending on scenario cost).
- **Cold vs warm**: `cold_ms` is the very first invocation, including all lazy
  setup. The tabulated `mean`/`median`/percentiles are warm statistics computed
  over the measured iterations only. Cold and warm are therefore reported
  separately, matching docs/performance/README.md.
- **Timing**: wall-clock milliseconds via `time.perf_counter`.
- **Percentiles**: linear interpolation at rank `(n - 1) * p / 100` — the same
  definition as `statistics.quantiles(method="inclusive")` and numpy's default.
  P50 equals the median; P0/P100 are min/max. Mean, median, min, max and
  (sample) standard deviation are recorded alongside P50/P90/P95/P99.
- **RAM (`peak_ram_bytes`)**: measured with `tracemalloc` across all runs of a
  scenario. **Caveat:** tracemalloc tracks *Python-level allocations only* —
  memory allocated by native extensions (ONNX runtime buffers, model weights,
  qdrant-client native storage, C-extension BM25 buffers) is invisible, so the
  figure is a lower bound on real process memory. VRAM is not measured here.
- **Determinism**: all synthetic documents/chunks are generated from a fixed
  word bank (`support.py`), so workloads are byte-identical across machines and
  report files; mock embedders are hash-based and deterministic.

## Reports

At session end, `conftest.py` flushes the collected results (all scenarios
share one session-scoped `ReportWriter`) into:

```
benchmarks/results/<UTC timestamp>-rag-aio-benchmarks.json   # full per-iteration data
benchmarks/results/<UTC timestamp>-rag-aio-benchmarks.md     # human-readable table
```

The Markdown table is sorted by benchmark name; the JSON file additionally
contains every individual iteration timing. Both carry an environment header
recording Python version, platform, CPU count, timestamp, and a config hash.

`benchmarks/results/` is git-ignored. When publishing results per release
(docs/performance/README.md), attach the generated files **together with the
hardware description** (CPU model, RAM, OS, accelerator if any) — the report
header alone does not identify the machine. Numbers from different hardware or
different `config_hash` values must not be compared directly.
