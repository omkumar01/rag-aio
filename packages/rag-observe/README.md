# rag-observe

OpenTelemetry API-based instrumentation for every rag-aio stage (ingestion, parsing, OCR,
chunking, embedding, retrieval, reranking, context, generation), plus structured logging
with default redaction of secrets and document content. Depends on the OTel API only;
exporters are optional extras.
