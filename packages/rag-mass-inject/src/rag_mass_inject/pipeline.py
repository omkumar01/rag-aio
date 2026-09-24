"""Batch ingestion orchestrator: discover → process → checkpoint → dead-letter.

:class:`MassIngestor` wires the :class:`~rag_doc_handler.pipeline.IngestionPipeline`
(load + parse) and :class:`~rag_embedder.pipeline.EmbeddingPipeline` (chunk +
embed + index) behind a bounded, resumable job model.  Each job is tracked in
SQLite via :class:`~rag_mass_inject.jobs.JobTracker`; per-stage concurrency is
enforced by :class:`~rag_mass_inject.workers.StageWorker`.

``rag_orchestrator`` is imported **only under ``TYPE_CHECKING``** — the
ingestor accepts any object structurally compatible with
:class:`~rag_orchestrator.services.OrchestratorServices`, keeping the runtime
dependency surface to ``rag-core``, ``rag-doc-handler`` and ``rag-embedder``.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rag_core.documents import Document
from rag_core.errors import ConfigError, JobError, UnsupportedFormatError
from rag_core.jobs import JobStatus, PipelineJob
from rag_doc_handler.pipeline import DedupIndex

from .config import MassInjectConfig
from .jobs import JobTracker
from .sources import discover_directory, discover_file_list, discover_sitemap, is_supported
from .workers import BoundedQueue, StageWorker

if TYPE_CHECKING:
    from rag_orchestrator.services import OrchestratorServices

__all__ = ["MassIngestor"]


# A sentinel pushed onto the bounded queue to tell a worker coroutine to stop.
_SHUTDOWN: Any = None


class MassIngestor:
    """High-throughput batch ingestion orchestrator.

    Parameters
    ----------
    config:
        Tuning knobs (concurrency, checkpoint dir, resume flag, …).
    services:
        A bag of wired components (duck-typed against
        :class:`~rag_orchestrator.services.OrchestratorServices`).  Must
        provide ``ingestion_pipeline``, ``embedding_pipeline``,
        ``vector_store`` and ``observer``.
    """

    def __init__(self, config: MassInjectConfig, services: OrchestratorServices) -> None:
        self.config = config
        self.services = services
        self.tracker = JobTracker(config.checkpoint_dir)
        self._worker = StageWorker(config)

    # -- public API -----------------------------------------------------------

    def submit(self, source: str) -> str:
        """Create a queued job record for *source* and return its id.

        This is **synchronous** so callers (including sync CLI wrappers) can
        obtain a job id immediately.  The actual ingestion happens in
        :meth:`wait`.
        """
        return self.tracker.create(source, kind="mass_ingest")

    async def status(self, job_id: str) -> PipelineJob:
        """Return the current :class:`PipelineJob` record for *job_id*.

        Raises :class:`~rag_core.errors.JobError` if the job does not exist.
        """
        job = await self.tracker.get(job_id)
        if job is None:
            msg = f"job not found: {job_id}"
            raise JobError(msg)
        return job

    async def wait(self, job_id: str) -> list[Document]:
        """Run the job to completion and return the ingested documents.

        Respects ``config.timeout_s`` via :func:`asyncio.timeout`.  On
        timeout the job is marked ``failed`` and an
        :class:`~rag_core.errors.OperationTimeout` is raised.
        """
        job = await self.tracker.get(job_id)
        if job is None:
            msg = f"job not found: {job_id}"
            raise JobError(msg)
        source = job.result_summary.get("source", "")
        if not source:
            msg = f"job {job_id} has no source"
            raise JobError(msg)

        try:
            async with asyncio.timeout(self.config.timeout_s):
                docs = await self._run_job(job_id, source)
            return docs
        except TimeoutError as exc:
            await self.tracker.update(
                job_id,
                status=JobStatus.failed,
                error=f"timeout after {self.config.timeout_s}s",
                finished_at=datetime.now(UTC),
            )
            from rag_core.errors import OperationTimeout

            raise OperationTimeout(
                f"mass inject exceeded {self.config.timeout_s}s",
                details={"timeout_s": self.config.timeout_s, "job_id": job_id},
            ) from exc
        except Exception as exc:
            await self.tracker.update(
                job_id,
                status=JobStatus.failed,
                error=str(exc),
                finished_at=datetime.now(UTC),
            )
            raise

    # -- job execution --------------------------------------------------------

    async def _run_job(self, job_id: str, source: str) -> list[Document]:
        """Discover files from *source* and process every file concurrently.

        Uses a :class:`BoundedQueue` with ``max_workers`` consumer coroutines.
        Each consumer calls :meth:`_process_file` which itself is gated by
        per-stage semaphores from :class:`StageWorker`.
        """
        files = await self._discover(source)
        total = len(files)
        started_at = datetime.now(UTC)
        await self.tracker.update(
            job_id,
            status=JobStatus.running,
            started_at=started_at,
            total_items=total,
            processed_items=0,
            progress=0.0,
        )
        self.services.observer.record(
            "mass_inject.start",
            {"job_id": job_id, "source": source, "total_items": total},
        )

        if total == 0:
            await self.tracker.update(
                job_id,
                status=JobStatus.completed,
                finished_at=datetime.now(UTC),
                progress=1.0,
                processed_items=0,
            )
            return []

        dedup_index = DedupIndex()
        documents: list[Document] = []
        processed = 0
        queue: BoundedQueue = BoundedQueue(maxsize=self.config.max_workers * 2)

        async def _worker() -> None:
            nonlocal processed
            while True:
                item = await queue.get()
                if item is _SHUTDOWN:
                    queue.task_done()
                    return
                try:
                    doc = await self._process_file(str(item), job_id, dedup_index)
                    if doc is not None:
                        documents.append(doc)
                finally:
                    processed += 1
                    progress = min(processed / total, 1.0) if total else 1.0
                    await self.tracker.update(
                        job_id,
                        processed_items=processed,
                        progress=progress,
                    )
                    queue.task_done()

        consumers = [asyncio.create_task(_worker()) for _ in range(self.config.max_workers)]

        # Producer: enqueue every discovered file.
        for f in files:
            await queue.put(f)
        # Send one shutdown sentinel per consumer.
        for _ in range(self.config.max_workers):
            await queue.put(_SHUTDOWN)

        await asyncio.gather(*consumers)

        await self.tracker.update(
            job_id,
            status=JobStatus.completed,
            finished_at=datetime.now(UTC),
            progress=1.0,
            processed_items=processed,
        )
        self.services.observer.record(
            "mass_inject.complete",
            {"job_id": job_id, "documents": len(documents), "processed": processed},
        )
        return documents

    async def _discover(self, source: str) -> list[str]:
        """Turn a *source* (directory / file / sitemap URL / file-list) into paths."""
        if source.startswith(("http://", "https://")):
            return await discover_sitemap(source)
        # Filesystem operations are blocking — offload to a worker thread.
        return await asyncio.to_thread(self._discover_sync, source)

    def _discover_sync(self, source: str) -> list[str]:
        """Synchronous path discovery (offloaded to avoid blocking the loop)."""
        path = Path(source)
        if path.is_dir():
            return [str(p) for p in discover_directory(source, recursive=self.config.recursive)]
        if path.is_file():
            if is_supported(source):
                return [source]
            # Not a supported document — might be a file-list text file.
            try:
                return discover_file_list(path)
            except (OSError, UnicodeDecodeError):
                return [source]
        return []

    async def _process_file(
        self,
        source: str,
        job_id: str,
        dedup_index: DedupIndex,
    ) -> Document | None:
        """Process a single file: load → parse → chunk → embed → index.

        Returns the ingested :class:`Document` on success, or ``None`` when
        the file is skipped (duplicate within run, or already checkpointed
        under ``resume=True``) or dead-lettered.
        """
        pipeline = self.services.ingestion_pipeline
        if pipeline is None:
            msg = "ingestion_pipeline is required: wire one via load_local_services()"
            raise ConfigError(msg)
        embeddings = self.services.embedding_pipeline
        if embeddings is None:
            msg = "embedding_pipeline is required: wire one via load_local_services()"
            raise ConfigError(msg)
        store = self.services.vector_store
        if store is None:
            msg = "vector_store is required: wire one via load_local_services()"
            raise ConfigError(msg)

        try:
            # --- load + parse + within-run dedup --- #
            result: tuple[Document, bool] = cast(
                "tuple[Document, bool]",
                await self._worker.run_stage("parse", pipeline.ingest(source, dedup=dedup_index)),
            )
            document, is_new = result

            # Duplicate content within this run — skip embedding.
            if not is_new:
                self.services.observer.record(
                    "mass_inject.dedup_skip",
                    {"source": source, "content_hash": document.content_hash},
                )
                return None

            # Resume: skip files already checkpointed in a prior run.
            if self.config.resume and await self.tracker.is_checkpointed(
                job_id, document.content_hash
            ):
                self.services.observer.record(
                    "mass_inject.resume_skip",
                    {"source": source, "content_hash": document.content_hash},
                )
                return None

            # --- chunk + embed + index --- #
            outcome = await self._worker.run_stage("embed", embeddings.process(document, store))

            # Record checkpoint so a future resume skips this file.
            await self.tracker.checkpoint(job_id, source, document.content_hash)

            self.services.observer.record(
                "mass_inject.ingested",
                {
                    "source": source,
                    "document_id": document.id,
                    "chunks_indexed": outcome.chunks_indexed,
                    "took_ms": outcome.took_ms,
                },
            )
            return document

        except UnsupportedFormatError as exc:
            await self.tracker.dead_letter(job_id, source, str(exc), exc.code)
            self._copy_to_dead_letter(source)
            self.services.observer.record(
                "mass_inject.dead_letter",
                {"source": source, "error": str(exc), "code": exc.code},
            )
            return None
        except Exception as exc:
            await self.tracker.dead_letter(job_id, source, str(exc))
            self._copy_to_dead_letter(source)
            self.services.observer.record(
                "mass_inject.dead_letter",
                {"source": source, "error": str(exc), "code": type(exc).__name__},
            )
            return None

    # -- helpers --------------------------------------------------------------

    def _copy_to_dead_letter(self, source: str) -> None:
        """Best-effort copy of *source* into ``config.dead_letter_dir``."""
        try:
            dl_dir = Path(self.config.dead_letter_dir)
            dl_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(source)
            if src_path.is_file():
                dest = dl_dir / src_path.name
                shutil.copy2(str(src_path), str(dest))
        except Exception:
            pass
