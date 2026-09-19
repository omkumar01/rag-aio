# rag-ui

Streamlit management and experimentation console for rag-aio. Every non-secret runtime
setting is overridable and validated with the same Pydantic models the backend uses
(presets, save/load, versioning, reset, import/export; secrets masked). Dashboards cover
ingestion/jobs/documents/chunk statistics/retrieval/reranker/citations/traces/evaluation/
cache/models/health, plus the flagship diagnostic workflow: select a document and inspect
its complete pipeline from original file to final answer with citations.
