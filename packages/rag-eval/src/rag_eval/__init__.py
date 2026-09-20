"""rag-eval: first-class evaluation framework for rag-aio.

Provides retrieval ranking metrics, generation quality heuristics, LLM-judge
adapters, dataset loaders (JSONL/CSV/Parquet), an async evaluation runner,
and report formatting.
"""

from .datasets import (
    EvalCase,
    load_csv,
    load_jsonl,
    load_parquet,
    save_jsonl,
)
from .generation_metrics import (
    JudgeVerdict,
    LLMJudge,
    OpenAICompatibleJudge,
    answer_overlap,
    citation_precision,
    citation_recall,
    faithfulness_proxy,
)
from .metrics import (
    METRICS,
    MetricFunc,
    average_precision,
    compute_metric,
    hit_rate,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)
from .report import format_report
from .runner import EvalRunner, compare

__version__ = "0.1.0"

__all__ = [
    # ranking metrics
    "METRICS",
    # datasets
    "EvalCase",
    # runner & report
    "EvalRunner",
    # generation metrics
    "JudgeVerdict",
    "LLMJudge",
    "MetricFunc",
    "OpenAICompatibleJudge",
    "__version__",
    "answer_overlap",
    "average_precision",
    "citation_precision",
    "citation_recall",
    "compare",
    "compute_metric",
    "faithfulness_proxy",
    "format_report",
    "hit_rate",
    "load_csv",
    "load_jsonl",
    "load_parquet",
    "mrr",
    "ndcg",
    "precision_at_k",
    "recall_at_k",
    "save_jsonl",
]
