"""FastAPI service surface for the rag-orchestrator.

This module is only imported when a service host is requested (via
:func:`rag_orchestrator.create_app`), so ``import rag_orchestrator`` itself does
not require FastAPI to be installed.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from fastapi import FastAPI
from pydantic import Field  # noqa: F401  (re-exported below / reserved for models)
from rag_core.base import RagBaseModel
from rag_core.ids import new_id
from rag_core.jobs import JobStatus, PipelineJob

from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.ingest import ingest as ingest_one
from rag_orchestrator.orchestrator import AskResult, Orchestrator
from rag_orchestrator.services import OrchestratorServices

__all__ = ["AskRequest", "IngestRequest", "create_app"]


class AskRequest(RagBaseModel):
    query: str
    stream: bool = False
    overrides: dict[str, Any] | None = None
    query_id: str | None = None
    correlation_id: str | None = None


class IngestRequest(RagBaseModel):
    source: str
    recursive: bool = False


def create_app(
    services: OrchestratorServices,
    pipeline_config: PipelineConfig | None = None,
) -> FastAPI:
    from rag_orchestrator import __version__, pipeline_schema_version
    from rag_orchestrator.pipeline import default_registry

    app = FastAPI(
        title="rag-orchestrator",
        version=__version__,
        description="Strategy/workflow orchestration API for rag-aio.",
    )
    orchestrator = Orchestrator(services, pipeline_config)
    app.state.orchestrator = orchestrator
    app.state.services = services
    jobs: dict[str, PipelineJob] = {}
    app.state.jobs = jobs
    app.state.registry = default_registry()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        checks: dict[str, Any] = {"ready": True, "components": {}}
        store = services.vector_store
        if store is not None:
            health_fn = getattr(store, "health", None)
            if callable(health_fn):
                checks["components"]["vector_store"] = await health_fn()
            else:
                checks["components"]["vector_store"] = True
        doc_store = services.document_store
        if doc_store is not None:
            dh = getattr(doc_store, "health", None)
            if callable(dh):
                checks["components"]["document_store"] = await dh()
        if services.cache is not None:
            checks["components"]["cache"] = True
        checks["ready"] = all(isinstance(v, bool) and v for v in checks["components"].values())
        return checks

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        store = services.vector_store
        cache = services.cache
        count_val: int | None = None
        count_fn = getattr(store, "count", None)
        if callable(count_fn):
            try:
                # InMemoryVectorStore.count is sync; Qdrant/SQL stores are async.
                outcome = count_fn()
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                count_val = int(outcome or 0)
            except (TypeError, ValueError):
                count_val = None
        cache_stats: dict[str, Any] = {}
        stats_raw = getattr(cache, "stats", None)
        # InstrumentedCache exposes .stats as a property returning CacheStats;
        # other backends may expose a stats() callable returning a dict.
        if callable(stats_raw):
            stats_raw = stats_raw()
        if isinstance(stats_raw, dict):
            cache_stats = stats_raw
        elif stats_raw is not None and hasattr(stats_raw, "hits"):
            cache_stats = {
                "hits": stats_raw.hits,
                "misses": stats_raw.misses,
                "sets": stats_raw.sets,
                "deletes": stats_raw.deletes,
                "evictions": stats_raw.evictions,
                "errors": stats_raw.errors,
                "hit_rate": stats_raw.hit_rate,
            }
        return {
            "pipeline": orchestrator.config.name,
            "schema_version": pipeline_schema_version,
            "vector_points": count_val,
            "cache": cache_stats,
        }

    @app.get("/version")
    async def version() -> dict[str, Any]:
        return {
            "version": __version__,
            "schema_version": pipeline_schema_version,
            "pipeline": orchestrator.config.name,
        }

    @app.get("/v1/pipelines")
    async def list_pipelines() -> dict[str, Any]:
        registry = app.state.registry
        return {
            "schema_version": pipeline_schema_version,
            "pipelines": [p.to_dict() for p in registry.list()],
        }

    @app.post("/v1/ask")
    async def ask_endpoint(body: AskRequest) -> Any:
        if body.stream:

            async def events() -> Any:
                stream = await orchestrator.ask(
                    body.query,
                    stream=True,
                    overrides=body.overrides,
                    query_id=body.query_id,
                    correlation_id=body.correlation_id,
                )
                assert not isinstance(stream, AskResult)  # stream=True path
                async for delta in stream:
                    yield f"data: {json.dumps(delta)}\n\n"
                yield "data: [DONE]\n\n"

            from fastapi.responses import StreamingResponse

            return StreamingResponse(events(), media_type="text/event-stream")

        result = await orchestrator.ask(
            body.query,
            overrides=body.overrides,
            query_id=body.query_id,
            correlation_id=body.correlation_id,
        )
        assert isinstance(result, AskResult)
        from fastapi.responses import JSONResponse

        return JSONResponse(result.model_dump(mode="json"))

    @app.post("/v1/ingest")
    async def ingest_endpoint(body: IngestRequest) -> Any:
        job_id = new_id()
        try:
            document = await ingest_one(services, body.source, config=orchestrator.config)
        except Exception as exc:
            job = PipelineJob(
                kind="ingest",
                status=JobStatus.failed,
                stage="index",
                error=str(exc),
                error_code=getattr(exc, "code", "ingestion_error"),
            )
            app.state.jobs[job_id] = job
            from fastapi import HTTPException
            from fastapi import status as _status

            raise HTTPException(
                status_code=_status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"job_id": job_id, "error": str(exc)},
            ) from exc
        job = PipelineJob(
            kind="ingest",
            status=JobStatus.completed,
            stage="done",
            total_items=1,
            processed_items=1,
            result_summary={
                "document_id": document.id,
                "content_hash": document.content_hash,
                "pages": len(document.pages),
            },
        )
        app.state.jobs[job_id] = job
        from fastapi.responses import JSONResponse

        return JSONResponse(document.model_dump(mode="json"))

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> PipelineJob:
        job = jobs.get(job_id)
        if job is None:
            from fastapi import HTTPException
            from fastapi import status as _status

            raise HTTPException(
                status_code=_status.HTTP_404_NOT_FOUND,
                detail=f"job {job_id} not found",
            )
        return job

    return app
