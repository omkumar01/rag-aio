# rag-eval

First-class evaluation framework for the rag-aio platform.

## Metrics

**Ranking metrics** (pure-Python, no ranx/trec dependencies):

| Metric | Description |
|--------|-------------|
| `recall_at_k` | Recall@k: fraction of relevant docs in top-k |
| `precision_at_k` | Precision@k: fraction of top-k that are relevant |
| `hit_rate` | 1.0 if any relevant doc appears in top-k |
| `mrr` | Mean Reciprocal Rank of first relevant doc |
| `average_precision` | Mean of precision at each relevant hit |
| `ndcg` | Normalized DCG (binary gains, IDCG with min(\|relevant\|, k) |

All functions share the signature `(ground_truth: list[str], predicted: list[str], k: int | None = None) -> float`.
Empty inputs return `0.0`.

Use `compute_metric("recall@10", gt, pred)` to parse `@k` syntax, or access
the `METRICS` registry directly.

## Generation Quality

- `citation_precision` / `citation_recall` - citation correctness
- `answer_overlap` - Jaccard token-set overlap (groundedness proxy)
- `faithfulness_proxy` - fraction of 4-gram shingles from the answer present in context
- `LLMJudge` / `OpenAICompatibleJudge` - LLM-as-judge adapter wrapping any duck-typed
  generator with `async generate(GenerationRequest) -> GenerationResult`; includes
  robust JSON extraction and score clamping.

## Datasets

- `load_jsonl` / `save_jsonl` - JSONL with per-line validation
- `load_csv` - CSV with `;`-separated `relevant_ids`
- `load_parquet` - Parquet via pyarrow (import-guarded; raises `EvaluationError`
  with install hint if pyarrow is missing)

All loaders validate against `EvalCase` and raise `EvaluationError` with row/line
numbers on invalid data.

## Runner

```python
from rag_eval import EvalRunner


async def my_retriever(case):
    result = await some_retrieval(case.query)
    return result  # must have .hits with .chunk_id


runner = EvalRunner(retriever_factory=my_retriever, metrics=["recall@10", "mrr", "ndcg@10"])
result = await runner.run(cases)
report = format_report(result)
```

The runner records `latency_ms` per case, classifies retrieval failures as
`retrieval_error`, computes aggregated metrics over successful cases only, and
raises `EvaluationError` if all cases fail.
