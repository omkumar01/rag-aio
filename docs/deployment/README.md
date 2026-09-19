# Deployment

## Topologies

1. **In-process (default)**: the whole platform embedded in one Python process — lowest
   latency, zero infrastructure. Qdrant local mode + sqlite.
2. **Workstation**: in-process platform + LM Studio/vLLM/Ollama for models.
3. **Scaled services**: selected modules (doc-handler, ocr, embedder, retrieval, rerank,
   generation) deployed as FastAPI services behind the orchestrator; Qdrant server,
   PostgreSQL, Redis for shared state. See `docker-compose.qdrant.yml` for the vector
   store.

Service topology guidance per module (when it pays off, resource sizing) is documented
here as services land.
