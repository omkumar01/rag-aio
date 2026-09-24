# rag-mass-inject

High-throughput bulk ingestion for rag-aio: discover sources from directories,
file lists, sitemaps, or object-storage-style paths; stream them through
load -> parse -> chunk -> embed -> index; and persist job state with SQLite
checkpoints so runs can be resumed. Every stage is bounded by per-stage
semaphores, backpressure is enforced by a bounded queue (never unbounded RAM),
and failures are dead-lettered (copied aside + recorded) rather than aborting
the whole job.

`rag-mass-inject` does **not** implement loading, parsing, chunking, embedding,
or indexing itself — it composes the pipelines from `rag-doc-handler` and
`rag-embedder`. It accepts any object structurally compatible with
`OrchestratorServices` (duck-typed), so the only *runtime* dependencies are
`rag-core`, `rag-doc-handler`, and `rag-embedder`; the concrete
`OrchestratorServices` type is referenced only under `TYPE_CHECKING`.

## Installation

```bash
uv add -e packages/rag-mass-inject
```

This pulls in `rag-core`, `rag-doc-handler`, `rag-embedder`, `rag-ocr`,
`rag-db-handler`, `httpx>=0.27`, and `aiosqlite>=0.20`. To drive the ingestor
end-to-end against real backends, install the orchestrator and its collaborators
as well (this is what the workspace installs by default):

```bash
uv add -e packages/rag-mass-inject \
  rag-orchestrator rag-cache rag-retrieval rag-rerank rag-context \
  rag-query rag-llm-provider
```

The `[test]` extra declares the lightweight mock/test collaborators used by the
test suite.

## Architecture / Design Principles

`MassIngestor` is a **bounded, resumable batch orchestrator** that bridges two
existing pipelines:

```
discover  ──▶  ingestion_pipeline (rag-doc-handler)
  │              load ─▶ detect ─▶ parse ─▶ dedup
  │
sources      chunk ─▶ embed ─▶ index  (rag-embedder)
  │              └─▶ vector_store (duck-typed)
  ▼
job_id  ──▶  JobTracker (SQLite)
```

**Key design decisions:**

- **Producer/consumer with backpressure.** Discovered file paths are pushed onto
  a `BoundedQueue(maxsize=max_workers * 2)`. A pool of `max_workers` consumer
  coroutines drains the queue, so memory usage never grows with the source set.
- **Per-stage concurrency.** `StageWorker` gives each pipeline stage (`read`,
  `parse`, `ocr`, `chunk`, `embed`, `index`) its own `asyncio.Semaphore`, so
  the lightweight "read" stage can run more concurrent jobs than the heavier
  "embed" stage. Stages not listed in `concurrency` run unbounded.
- **Resumable via SQLite.** `JobTracker` stores one row per processed file
  (a *checkpoint*), keyed by `(job_id, content_hash)`. The check uses
  `INSERT OR IGNORE`, so concurrent workers never create duplicate checkpoint
  rows. On resume, files whose `content_hash` already has a checkpoint are
  skipped — unchanged documents are never re-indexed.
- **Two layers of dedup.** Within a single run, `DedupIndex` (from
  `rag-doc-handler`) catches identical source-level duplicates. Across runs,
  the SQLite checkpoint store catches content that was already ingested.
  `Document.content_hash` is SHA-256 over `source_uri + normalized text`, so
  the same file at the same path always maps to the same hash.
- **Dead-letter, never abort.** `UnsupportedFormatError` and generic
  exceptions are caught per-file. The offending file is recorded in the
  `dead_letters` table and copied into `config.dead_letter_dir`. The job
  itself completes successfully (status `completed`), so a single corrupt file
  never poisons a million-file run.
- **Sync `submit` / async `wait`.** `submit()` is synchronous and returns a
  job id immediately; `wait()` runs the job asynchronously to completion and
  returns the list of ingested `Document` objects. This split lets synchronous
  CLI callers obtain a handle before the async phase begins.
- **Observability.** Every significant event is recorded through the injected
  `OrchestratorServices.observer` — `mass_inject.start`, `dedup_skip`,
  `resume_skip`, `ingested`, `dead_letter`, `mass_inject.complete` — so a
  `rag-observe` tracer downstream sees per-file and per-job signals.

## Public API

```python
from rag_mass_inject import (
    # orchestrator
    MassIngestor,
    MassInjectConfig,
    # checkpoint store
    JobTracker,
    # concurrency primitives
    BoundedQueue,
    StageWorker,
    # source discovery
    discover_directory,
    discover_file_list,
    discover_sitemap,
    is_supported,
    SUPPORTED_EXTENSIONS,
    # version
    __version__,
)
```

### MassIngestor

```python
class MassIngestor(config: MassInjectConfig, services: OrchestratorServices):
    def submit(source: str) -> str                     # queue a job, return id (sync)
    async def wait(job_id: str) -> list[Document]      # run to completion, return docs
    async def status(job_id: str) -> PipelineJob       # current job record
```

`services` is any bag exposing `ingestion_pipeline`, `embedding_pipeline`,
`vector_store`, and `observer`. The canonical source is
`rag_orchestrator.load_local_services()`.

### MassInjectConfig

```python
MassInjectConfig(
    max_workers: int = 4,                       # concurrent file-processing consumers
    recursive: bool = False,                    # recurse into subdirectories
    concurrency: dict[str, int] = {             # per-stage semaphore limits
        "read": 4, "parse": 2, "ocr": 2,
        "chunk": 2, "embed": 4, "index": 4,
    },
    checkpoint_dir: str = "./data/mass-inject", # SQLite job/checkpoint DB lives here
    dead_letter_dir: str = "./data/mass-inject/dead-letter",
    resume: bool = True,                        # skip files with existing checkpoints
    timeout_s: float | None = None,             # end-to-end wait() timeout
)
```

`.stage_limit(stage) -> int` returns the configured semaphore capacity for a
stage (default 1 when absent). `.local_default()` returns a local-development
profile. `.dump_for_log()` returns a log-safe JSON dict.

### JobTracker (SQLite checkpoint store)

```python
tracker = JobTracker(checkpoint_dir)

job_id = tracker.create(source="/data/docs", kind="mass_ingest")       # sync
job: PipelineJob | None = await tracker.get(job_id)                     # async
jobs: list[PipelineJob] = await tracker.list()
await tracker.update(job_id, status=JobStatus.running, progress=0.5)
await tracker.checkpoint(job_id, source="/data/docs/a.pdf", content_hash="abc123")
await tracker.is_checkpointed(job_id, "abc123")   # -> True
await tracker.dead_letter(job_id, "/data/docs/bad.bin", "cannot parse", "unsupported_format")
```

The checkpoint row uses `INSERT OR IGNORE` on the primary key
`(job_id, content_hash)` — concurrent writers are safe; the first wins.

### BoundedQueue

```python
queue = BoundedQueue(maxsize=8)
await queue.put(item)     # blocks if full → backpressure
item = await queue.get()  # blocks if empty
queue.task_done()
await queue.join()
```

### StageWorker

```python
worker = StageWorker(config)
result = await worker.run_stage("embed", embeddings.process(document, store))
worker.semaphore("embed")._value  # current permit count (for assertions)
```

Stages absent from `config.concurrency` return `None` from `semaphore()` and
run without a gate.

### Source discovery

```python
# Local directory (flat or recursive)
files = discover_directory("/data/reports", recursive=True)   # list[Path]

# File-list text file (one path/URL per line; '#' comments and blanks skipped)
paths = discover_file_list("inputs.list")                       # list[str]

# XML sitemap → list of <loc> URLs
urls = await discover_sitemap("https://example.com/sitemap.xml")

# Extension check
is_supported("doc.pdf")   # True
is_supported("doc.bin")   # False
```

`SUPPORTED_EXTENSIONS` is a frozenset of `.pdf`, `.txt`, `.md`, `.html`,
`.htm`, `.docx`, `.xlsx`, `.pptx`, `.eml`.

## Usage Guides

### Beginner — ingest a local directory with default services

```python
import asyncio
from rag_aio.config import RAGConfig
from rag_aio.facade import build_services
from rag_mass_inject import MassIngestor, MassInjectConfig

async def main():
    services = build_services(RAGConfig.mock())    # fully offline mock backends
    config = MassInjectConfig.local_default()
    ingestor = MassIngestor(config, services)

    job_id = ingestor.submit("/path/to/documents")
    print(f"submitted job {job_id}")

    docs = await ingestor.wait(job_id)
    print(f"ingested {len(docs)} documents")

    job = await ingestor.status(job_id)
    print(f"status={job.status} progress={job.progress}")

asyncio.run(main())
```

### Intermediate — resume a previous run

With `resume=True` (the default), files whose content hash already has a
checkpoint are skipped on subsequent `wait()` calls for the *same* job id:

```python
config = MassInjectConfig(
    checkpoint_dir="./checkpoints",
    dead_letter_dir="./dead-letters",
    resume=True,
    max_workers=8,
)

ingestor = MassIngestor(config, services)
job_id = ingestor.submit("/data/large_corpus")
docs = await ingestor.wait(job_id)          # first pass: N documents

# Later — re-run the same job id; only new/changed files are processed.
docs_again = await ingestor.wait(job_id)     # 0 documents (all checkpointed)
```

### Advanced — tune per-stage concurrency and timeouts

```python
config = MassInjectConfig(
    max_workers=16,
    recursive=True,
    concurrency={
        "read": 16,    # I/O-bound, fan out
        "parse": 4,    # CPU-bound (pymupdf)
        "ocr": 2,      # GPU/heavy
        "chunk": 8,
        "embed": 4,    # model-bound
        "index": 8,    # DB writes
    },
    timeout_s=3600.0,  # fail the job after 1 hour
    resume=True,
)
```

### Advanced — custom source discovery

Supply a sitemap URL directly — `_discover` detects `http://` / `https://`
prefixes and routes to `discover_sitemap`:

```python
job_id = ingestor.submit("https://docs.example.com/sitemap.xml")
docs = await ingestor.wait(job_id)
```

Or use a file-list for a hand-curated set of heterogeneous sources:

```python
# inputs.list:
#   # my sources
#   /data/report.pdf
#   https://example.com/page.html
#   /data/notes.txt
job_id = ingestor.submit("inputs.list")
```

### Advanced — inspecting dead letters and checkpoints programmatically

```python
import aiosqlite

tracker = ingestor.tracker
async with aiosqlite.connect(tracker._db_path) as db:
    db.row_factory = aiosqlite.Row
    rows = await db.execute(
        "SELECT source, error, error_code, ts FROM dead_letters WHERE job_id = ?",
        (job_id,),
    )
    dead = [dict(r) async for r in rows]

for entry in dead:
    print(f"dead-lettered {entry['source']}: {entry['error']}")
```

## Configuration

`MassInjectConfig` is a strict Pydantic model (via `RagBaseModel`,
`extra="forbid"`) so unknown keys fail loudly. All fields have sensible
defaults for local development.

| Field               | Default                          | Purpose                                           |
|---------------------|----------------------------------|---------------------------------------------------|
| `max_workers`       | `4`                              | Number of concurrent consumer coroutines.         |
| `recursive`         | `False`                          | Recurse into subdirectories when discovering.     |
| `concurrency`       | 6-stage dict (see above)         | Per-stage `asyncio.Semaphore` capacities.         |
| `checkpoint_dir`    | `"./data/mass-inject"`           | Directory holding `jobs.db`.                      |
| `dead_letter_dir`   | `"./data/mass-inject/dead-letter"` | Where poison files are copied.                  |
| `resume`            | `True`                           | Skip files with existing checkpoints.             |
| `timeout_s`         | `None`                           | End-to-end `wait()` timeout in seconds.           |

The checkpoint DB schema has three tables — `jobs`, `checkpoints`, and
`dead_letters` — created synchronously on `JobTracker.__init__` so the schema
is ready before any async call.

## Testing

All tests use `rag_orchestrator`'s test extras (`rag-cache`, `rag-retrieval`,
etc.) and the offline `mass_inject_fakes.make_services()` helper, which wires
`MockEmbedder`, `MockSparseEmbedder`, `InMemoryVectorStore`, `MemoryCache`,
and `NoOpObserver` — no network or model downloads.

```bash
uv run pytest packages/rag-mass-inject -q
```

| Test file            | Scope    | What it covers                                           |
|----------------------|----------|----------------------------------------------------------|
| `test_smoke.py`      | unit     | Import, `__version__`, public exports.                  |
| `test_jobs.py`       | unit     | `JobTracker` CRUD, checkpoint idempotency, isolation.   |
| `test_sources.py`    | unit     | `discover_directory`, `discover_file_list`, sitemap parsing, `is_supported`. |
| `test_pipeline.py`   | integration | Full submit → wait → status flow, dedup, resume, dead-letter, recursive dirs. |

## Dependencies

| Dependency        | Role                                                     |
|-------------------|----------------------------------------------------------|
| `rag-core`        | `Document`, `PipelineJob`, `JobStatus`, error classes, `new_id`. |
| `rag-doc-handler` | `IngestionPipeline.ingest()` → `(Document, is_new)`, `DedupIndex`. |
| `rag-embedder`    | `EmbeddingPipeline.process(document, store)` → `PipelineOutcome`. |
| `rag-ocr`         | OCR fallback inside `PDFParser` (used via the ingestion pipeline). |
| `rag-db-handler`  | `VectorStore` implementations (duck-typed; e.g. `InMemoryVectorStore`). |
| `httpx>=0.27`     | Sitemap fetching (with injectable `AsyncClient` for tests). |
| `aiosqlite>=0.20` | Async SQLite access for `JobTracker`.                     |

**Optional / test only:** `rag-cache`, `rag-retrieval`, `rag-rerank`,
`rag-context`, `rag-query`, `rag-llm-provider`, `rag-orchestrator` — needed to
construct a full `OrchestratorServices` bag via `load_local_services()`.

## Cross-Package Relationships

`rag-mass-inject` is the **bulk-ingestion entry point** into the rag-aio data
plane. It is intentionally thin — every real capability comes from a dependency:

- **rag-orchestrator** — provides `OrchestratorServices`, the typed bag that
  `MassIngestor` accepts. This dependency is **TYPE_CHECKING-only**: the
  ingestor duck-types against the bag (`services.ingestion_pipeline`,
  `.embedding_pipeline`, `.vector_store`, `.observer`), so `rag-orchestrator`
  is never imported at runtime unless the caller explicitly wires it. This
  keeps the dependency surface minimal.
- **rag-doc-handler** — `IngestionPipeline.ingest(source, dedup)` returns
  `(Document, is_new)`. The `DedupIndex` provides within-run dedup. Source
  discovery helpers (`discover_directory`, `discover_file_list`,
  `discover_sitemap`) live in `rag-mass-inject` itself.
- **rag-embedder** — `EmbeddingPipeline.process(document, store)` runs chunk →
  embed → sparse → index and returns a `PipelineOutcome` (chunks indexed, dims,
  model, elapsed ms).
- **rag-core** — `Document` carries `content_hash` (SHA-256 of source + text)
  used for checkpointing; `PipelineJob` and `JobStatus` are the job-model
  types persisted by `JobTracker`; error classes (`ConfigError`, `JobError`,
  `UnsupportedFormatError`, `OperationTimeout`) are re-raised.
- **rag-aio facade** — `RAG.from_config()` builds services via
  `build_services()`; a `MassIngestor` can be constructed from those services
  for bulk runs, then a separate `Orchestrator` handles per-document
  re-ingestion and querying.
- **rag-observe** — the `observer` on `OrchestratorServices` receives
  `mass_inject.*` events, connecting bulk ingestion to distributed traces.

See also [architecture.md](../../architecture.md) and [adr/ADR-0007-job-model.md](../../adr/)
for the job/cancellation semantics this package builds on.
