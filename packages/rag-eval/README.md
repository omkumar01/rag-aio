# rag-eval

First-class evaluation framework for rag-aio. Provides retrieval ranking
metrics, generation-quality heuristics, LLM-as-judge adapters, dataset loaders
(JSONL / CSV / Parquet), an async evaluation runner with per-case latency and
failure classification, and markdown report formatting.

All ranking metrics are pure-Python with **no `ranx` or `trec` dependencies** —
they share a single signature and return `0.0` on empty inputs. Generation
metrics are LLM-free heuristics. The LLM judge is an adapter that wraps any
duck-typed generator; rag-eval does not ship or require a specific provider.

## Installation

```bash
uv add -e packages/rag-eval
```

This pulls in `rag-core` and `numpy>=1.26`.

Optional — parquet dataset support (install `pyarrow` separately):

```bash
uv add -e packages/rag-eval[parquet]   # or: pip install pyarrow
```

## Architecture / Design Principles

```
                    ┌──────────────────────────────────────────┐
                    │              rag-eval                  │
                    │                                         │
    EvalCase ──▶  EvalRunner  ──▶  retriever_factory  ──▶  [your retriever]
        │            │                │
        │            │                └─→  hits (→ chunk_id)
        │            │
        │            ├── compute_metric("recall@10")     (metrics.py)
        │            ├── answer_overlap / faithfulness    (generation_metrics.py)
        │            ├── OpenAICompatibleJudge            (LLM-as-judge)
        │            └── format_report()                  (report.py)
        │
        └──  load_jsonl / load_csv / load_parquet  (datasets.py)
```

**Key design decisions:**

- **Uniform metric signature.** Every ranking metric follows
  `(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float`.
  Empty `ground_truth` or `predicted` returns `0.0` — there is nothing to score.
  This makes metrics trivially composable and safe to average over a run.
- **Registry + `@k` parsing.** The `METRICS` dict maps canonical names
  (`recall`, `precision`, `hit_rate`, `mrr`, `average_precision`, `ndcg`) to
  functions. `compute_metric("recall@10", ...)` parses the `@k` suffix so
  callers specify cutoffs declaratively (e.g. in config files or YAML).
- **k semantics.** For `recall`/`precision`/`hit_rate`/`ndcg`, `k` defaults to
  10. For `mrr` and `average_precision`, `k` is accepted for registry
  compatibility but has no effect (these are rank-aware over the full list).
- **Generation metrics are LLM-free.** `citation_precision`,
  `citation_recall`, `answer_overlap` (Jaccard token-set), and
  `faithfulness_proxy` (4-gram shingle coverage) provide fast, deterministic
  proxies for groundedness and citation correctness — useful for regression
  testing before invoking a judge.
- **LLM judge is an adapter, not a default.** `OpenAICompatibleJudge` wraps any
  object with `async generate(GenerationRequest) -> GenerationResult` — the same
  duck-typed `Generator` protocol used by the rest of rag-aio. It sends a
  structured system prompt with a JSON schema, then runs robust extraction
  (direct parse → markdown code block → brace matching) and **clamps** the score
  to `[0.0, 1.0]`. A malformed LLM response degrades to `score=0.0,
  label="error"` rather than raising.
- **Dataset validation.** `EvalCase` inherits `RagBaseModel` (`extra="forbid"`),
  so extra fields in a JSONL line, CSV row, or Parquet record raise
  `EvaluationError` with the offending line/row number. `save_jsonl` confines
  the write target to a `base_dir` (only the filename component is used), and
  `load_jsonl`/`load_csv`/`load_parquet` reject `..` path traversal via
  `_safe_dataset_path`.
- **Runner resilience.** `EvalRunner.run` classifies any retriever exception as
  `retrieval_error` (recording `latency_ms` for the failed case) rather than
  aborting the suite. Aggregated metrics are computed over **successful cases
  only**. If *every* case fails, `EvaluationError` is raised with a failure
  summary. `config_hash` (from `rag_core.ids`) is recorded so repeated runs with
  the same metrics + `k` are reproducible and diffable via `compare()`.
- **Dependency minimality.** Only `rag-core` (contracts, error taxonomy,
  models) and `numpy` are required. `pyarrow` is import-guarded.

## Public API

```python
from rag_eval import (
    # ranking metrics
    recall_at_k,
    precision_at_k,
    hit_rate,
    mrr,
    average_precision,
    ndcg,
    METRICS,
    MetricFunc,
    compute_metric,
    # generation metrics
    citation_precision,
    citation_recall,
    answer_overlap,
    faithfulness_proxy,
    JudgeVerdict,
    LLMJudge,
    OpenAICompatibleJudge,
    # datasets
    EvalCase,
    load_jsonl,
    save_jsonl,
    load_csv,
    load_parquet,
    # runner & report
    EvalRunner,
    compare,
    format_report,
    # version
    __version__,
)
```

### Ranking metrics

All share `(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float`.

| Metric            | Description                                                                  |
|-------------------|------------------------------------------------------------------------------|
| `recall_at_k`     | Fraction of relevant docs in the top-*k* predictions.                        |
| `precision_at_k`  | Fraction of top-*k* predictions that are relevant.                           |
| `hit_rate`        | `1.0` if any relevant doc appears in the top-*k*, else `0.0`.                  |
| `mrr`             | Reciprocal rank of the first relevant doc (ignores *k*).                     |
| `average_precision` | Mean of precision at each relevant hit (ignores *k*).                      |
| `ndcg`            | Normalized DCG with binary gains; IDCG over `min(\|relevant\|, k)`.          |

```python
from rag_eval import recall_at_k, compute_metric, METRICS

gt = ["doc-a", "doc-b"]
pred = ["doc-c", "doc-a", "doc-d", "doc-b"]

assert recall_at_k(gt, pred, k=3) == 0.5  # 1 of 2 relevant in top-3
assert recall_at_k(gt, pred, k=4) == 1.0  # 2 of 2 in top-4
assert compute_metric("recall@3", gt, pred) == 0.5
assert compute_metric("mrr", gt, pred) == 0.5  # first hit at rank 2
assert METRICS["ndcg"](gt, pred, k=10) == 0.6509  # via registry
```

`compute_metric("nonexistent", ...)` raises `EvaluationError(code="unknown_metric")`;
`compute_metric("recall@abc", ...)` raises `EvaluationError(code="invalid_k")`.

### Generation metrics

```python
from rag_eval import citation_precision, citation_recall, answer_overlap, faithfulness_proxy

# Citation correctness
citation_precision(["a", "b", "c"], {"a", "b"})  # 2/3 ≈ 0.667
citation_recall({"a", "b", "c"}, ["a", "b"])  # 2/3 ≈ 0.667

# Groundedness proxies (no LLM needed)
answer_overlap("the cat sat", ["the cat sat on the mat"])  # Jaccard ≈ 0.5
faithfulness_proxy("the cat sat on the mat", ["the dog sat on the mat"])  # 1/3
```

### LLM judge

```python
from rag_eval import OpenAICompatibleJudge

# `generator` is any object with `async generate(GenerationRequest) -> GenerationResult`
# — e.g. rag-generation's GenerationService or a provider adapter.
judge = OpenAICompatibleJudge(
    generator=your_generator,
    model="gpt-4o",
    temperature=0.0,
    max_tokens=1024,
)

verdict = await judge.judge(
    answer="RAG augments language models with retrieved context.",
    context="RAG stands for Retrieval-Augmented Generation...",
    question="What does RAG stand for?",
)
print(verdict.score, verdict.label, verdict.reasoning)
```

`LLMJudge` is a `Protocol` with `async judge(answer, context, question) -> JudgeVerdict`;
`OpenAICompatibleJudge` is the included implementation. Both return a `JudgeVerdict`
with `score` clamped to `[0.0, 1.0]`.

### Datasets

```python
from rag_eval import EvalCase, load_jsonl, save_jsonl, load_csv, load_parquet

case = EvalCase(
    query_id="q1",
    query="What is RAG?",
    relevant_ids=["doc-1", "doc-2"],
    golden_answer="Retrieval-Augmented Generation",
    metadata={"source": "manual"},
)

# JSONL round-trip (per-line validation; blank lines skipped)
save_jsonl([case], "out.jsonl", base_dir="/data/eval")  # filename confined to base_dir
cases = load_jsonl("out.jsonl")  # raises EvaluationError on bad line

# CSV: columns query_id, query, relevant_ids (; separated), golden_answer
cases = load_csv("cases.csv")

# Parquet: requires pyarrow, raises EvaluationError(code="parquet_not_installed") if missing
cases = load_parquet("cases.parquet")
```

### Runner

```python
import asyncio
from rag_eval import EvalRunner, format_report


async def my_retriever(case: EvalCase):
    """Return an object with a ``hits`` attribute; each hit has a ``chunk_id``."""
    result = await some_hybrid_retrieval(case.query, top_k=10)
    return result  # e.g. rag_core.retrieval.RetrievalResult


runner = EvalRunner(
    retriever_factory=my_retriever,
    metrics=["recall@10", "precision@10", "mrr", "ndcg@10", "hit_rate@10"],
    k=10,  # default cutoff for bare metric names
    dataset="my-corpus",
)

cases = load_jsonl("eval.jsonl")
result = await runner.run(cases)
print(format_report(result))
```

`result` is a `rag_core.evaluation.EvaluationResult` with:
- `run_id`, `dataset`, `config_hash`, `model_versions`
- `metrics` — aggregated (averaged over successful cases)
- `per_query` — per-case breakdown with `failure_class` and `latency_ms`
- `started_at`, `finished_at`

`compare(result_a, result_b)` returns `{name: delta}` (b − a) for every metric
in either result; metrics present in only one side use `0.0`.

## Usage Guides

### Beginner — score a single retrieval

```python
from rag_eval import recall_at_k, ndcg, mrr

relevant = ["chunk-3", "chunk-7"]
retrieved = ["chunk-1", "chunk-3", "chunk-9", "chunk-7", "chunk-2"]

print("Recall@10:", recall_at_k(relevant, retrieved))  # 1.0 (both found)
print("nDCG@10:  ", ndcg(relevant, retrieved, k=10))  # 0.65
print("MRR:      ", mrr(relevant, retrieved))  # 0.5 (first hit at rank 2)
```

### Beginner — quick dataset from Python objects

```python
from rag_eval import EvalCase, save_jsonl

cases = [
    EvalCase(
        query_id="q1",
        query="What is RAG?",
        relevant_ids=["d1", "d2"],
        golden_answer="Retrieval-Augmented Generation",
    ),
    EvalCase(
        query_id="q2",
        query="What is BM25?",
        relevant_ids=["d3"],
        golden_answer="A sparse lexical retrieval algorithm",
    ),
]
save_jsonl(cases, "my_eval.jsonl", base_dir=".")
```

### Intermediate — run a full evaluation suite

```python
from rag_eval import EvalRunner, compute_metric


async def retrieve(case):
    # Plug in your RAG retriever — return anything with .hits (each .chunk_id)
    hits = await my_retriever.search(case.query, top_k=10)
    return type("R", (), {"hits": hits})()


runner = EvalRunner(
    retriever_factory=retrieve,
    metrics=["recall@10", "mrr", "ndcg@10"],
    dataset="production-qa",
)
result = await runner.run(load_jsonl("production-qa.jsonl"))

# Inspect per-query
for qe in result.per_query:
    if qe.failure_class:
        print(f"  {qe.query_id}: FAILED ({qe.failure_class})")
    else:
        scores = {m.name: round(m.value, 4) for m in qe.metrics}
        print(f"  {qe.query_id}: {scores}")
```

### Intermediate — LLM judge for answer faithfulness

```python
from rag_eval import OpenAICompatibleJudge


class MyGenerator:
    async def generate(self, request):
        # Your OpenAI-compatible provider here
        return await openai_client.chat(...)


judge = OpenAICompatibleJudge(generator=MyGenerator(), temperature=0.0)

cases = [
    EvalCase(
        query_id="q1",
        query="What is RAG?",
        relevant_ids=["d1"],
        golden_answer="...",
        metadata={"context": "RAG = ...", "answer": "RAG = ..."},
    ),
]
for case in cases:
    verdict = await judge.judge(
        answer=case.metadata["answer"],
        context=case.metadata["context"],
        question=case.query,
    )
    print(f"{case.query_id}: {verdict.label} ({verdict.score:.2f}) — {verdict.reasoning}")
```

### Advanced — compare two model configurations

```python
from rag_eval import EvalRunner, compare

runner_a = EvalRunner(retriever_factory=rag_a_retrieve, metrics=["recall@10", "ndcg@10"])
runner_b = EvalRunner(retriever_factory=rag_b_retrieve, metrics=["recall@10", "ndcg@10"])

result_a = await runner_a.run(cases)
result_b = await runner_b.run(cases)

deltas = compare(result_a, result_b)
for name, delta in deltas.items():
    flag = "up" if delta > 0 else "down" if delta < 0 else "same"
    print(f"  {name}: {delta:+.4f} ({flag})")
```

### Advanced — failure-resilient evaluation

If the retriever raises for some cases, the runner classifies them as
`retrieval_error`, records `latency_ms`, and excludes them from aggregation.
If **all** cases fail, `EvaluationError(code="all_retrieval_failed")` is raised:

```python
from rag_core.errors import EvaluationError

try:
    result = await runner.run(cases)
except EvaluationError as exc:
    print(f"all cases failed: {exc.details}")
```

## Configuration

`EvalRunner` is configured at construction:

| Parameter           | Default | Purpose                                                   |
|---------------------|---------|-----------------------------------------------------------|
| `retriever_factory` | —       | `Callable[[EvalCase], Awaitable[object with .hits]]`      |
| `metrics`           | built-in defaults | Metric names in `"base"` or `"base@k"` form.       |
| `k`                 | `10`    | Default cutoff for bare metric names (no `@k`).           |
| `dataset`           | `"rag-eval"` | Label recorded in `EvaluationResult.dataset`.          |

**Default metrics:** `recall@10`, `precision@10`, `mrr`, `ndcg@10`, `hit_rate@10`.

`OpenAICompatibleJudge` uses a fixed system prompt and a JSON schema for the
verdict output, defined as private module-level constants in
`generation_metrics.py` (`_JUDGE_SYSTEM_PROMPT`, `_JSON_SCHEMA`). To customize
judgment criteria, subclass `OpenAICompatibleJudge` and override `judge()`,
or build a verdict directly from `_parse_judge_response()` against your own
generator output.

## Testing

All tests are pure unit tests (marked `@pytest.mark.unit`) and require no
external services, models, or network.

```bash
uv run pytest packages/rag-eval -q
```

| Test file                   | Scope | What it covers                                                        |
|-----------------------------|-------|-----------------------------------------------------------------------|
| `test_smoke.py`             | unit  | Import, `__version__`, public exports.                               |
| `test_metrics.py`           | unit  | Every ranking metric + `compute_metric` `@k` parsing and errors.     |
| `test_generation_metrics.py`| unit  | Citation precision/recall, answer overlap, faithfulness, LLM judge JSON extraction + clamping. |
| `test_datasets.py`          | unit  | `EvalCase` validation, JSONL/CSV/Parquet round-trips, path traversal rejection. |
| `test_runner.py`            | unit  | `EvalRunner` metrics, latency, failure classification, `compare()`.  |
| `test_report.py`            | unit  | `format_report` — NaN/Inf handling, failure counts, 4-dp formatting.  |

## Dependencies

| Dependency   | Role                                                        |
|--------------|-------------------------------------------------------------|
| `rag-core`   | `EvaluationError`, `EvaluationResult`, `MetricResult`,      |
|              | `QueryEvaluation`, `GenerationRequest`, `GenerationResult`, |
|              | `Message`, `RagBaseModel`, `new_id`, `config_hash`.         |
| `numpy>=1.26` | Declared dependency; used by downstream embedding backends. |
| `pyarrow` (optional) | Parquet dataset loading. Import-guarded; raises         |
|              | `EvaluationError(code="parquet_not_installed")` if absent.  |

No `ranx`, `trec`, or other IR toolkit dependencies — all ranking metrics are
implemented from scratch in pure Python.

## Cross-Package Relationships

- **rag-core** — The bedrock. `EvaluationResult`, `MetricResult`, and
  `QueryEvaluation` live in `rag_core.evaluation` and are the canonical
  report models every result is built from. `EvalCase` extends
  `RagBaseModel` (`extra="forbid"`) for strict validation. `EvaluationError`
  carries the stable error taxonomy (`code` + `details`). `config_hash`
  (from `rag_core.ids`) fingerprints each run for reproducibility. `GenerationRequest`
  and `GenerationResult` are the contract the LLM judge adapter wraps.
- **rag-aio / rag-orchestrator** — The evaluation runner is designed to slot
  into the orchestrator's retrieval path: a `retriever_factory` receives an
  `EvalCase` and returns an object whose `.hits` carries `RetrievalHit`-compatible
  entries (each with a `chunk_id`). The runner's result model
  (`EvaluationResult`) is the same type the `rag_orchestrator` pipeline produces
  for job-status reporting, so eval results can be surfaced through the same
  FastAPI `/v1/jobs` and `/metrics` endpoints consumed by `rag-ui`.
- **rag-ui** — A future `Evaluation` dashboard page can render
  `format_report(result)` directly, or consume the structured
  `EvaluationResult` model for richer tables. The markdown report is
  human-readable by default (suitable for CI logs).

See also [architecture.md](../../architecture.md) and the `rag-core` error
taxonomy in [`rag_core/errors.py`](../../packages/rag-core/src/rag_core/errors.py).
