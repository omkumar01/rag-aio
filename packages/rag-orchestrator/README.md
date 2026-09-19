# rag-orchestrator

The strategy and workflow-management brain of rag-aio — not a monolithic RAG implementation.
Manages declarative pipeline definitions (parser → OCR policy → chunker → embedding →
index → retrieval → rerank → context → generation), named pipelines and versions, per-request
overrides, A/B hooks, fallback policies, timeout/cost/token budgets, and capability-based
routing. Structured concurrency with cancellation propagation; independent retrieval branches
run in parallel. Python API plus FastAPI service API (extra `fastapi`); PydanticAI (extra
`pydantic-ai`) only where agentic decisions genuinely help.
