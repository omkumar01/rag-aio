# rag-observe

OpenTelemetry **API**-based instrumentation and redacted structured logging for
every rag-aio stage (`rag-core` owns the `Observer` protocol; `rag-observe` is
the default implementation). Depends on the OTel API only — exporters are
optional extras and every call is a no-op when no SDK is configured.

## Public API

```python
from rag_observe import (
    ObservabilityHub,  # rag_core.protocols.Observer implementation
    observe,
    timed,  # span context managers
    StageTimer,  # dependency-free wall-clock timer
    record_counter,
    record_histogram,  # OTel metric helpers (no-op w/o SDK)
    get_logger,
    setup_logging,
    JSONFormatter,
    RedactingFilter,
    redact,
    REDACTED,
    truncate_for_log,
    set_correlation_id,
    get_correlation_id,
)
```

### Instrumentation (`instrumentation.py`)

* `observe(stage, attributes=None)` — sync context manager (usable inside `async
  def` bodies) that opens a `rag.<stage>` span, records `rag.stage` + your
  attributes + `rag.duration_ms`, and on exception records the exception and
  re-raises. No-op without an SDK.
* `timed(stage, attributes=None)` — like `observe` **plus** a
  `rag.<stage>.duration_ms` histogram.
* `record_counter(name, value, attributes=None)` / `record_histogram(...)` —
  thin wrappers over `opentelemetry.metrics.get_meter("rag-aio")`.
* `StageTimer` — `start()` / `stop()` / `.elapsed_ms`, usable without OTel.

```python
from rag_observe import observe, timed

with observe("retrieval", {"query_id": qid}):
    hits = await retriever.retrieve(query)

with timed("generation"):
    result = await generator.generate(request)
```

### Structured logging (`logging.py`)

* `setup_logging(level="INFO", json_output=True, log_document_content=False)` —
  configures the root logger once (idempotent; replaces prior managed handlers).
* `get_logger(name)` — returns a stdlib logger; emits JSON lines once
  `setup_logging` has run.
* `JSONFormatter` — one JSON object per line: `timestamp`, `level`, `logger`,
  `message`, `correlation_id`, `fields`.
* `set_correlation_id` / `get_correlation_id` — `contextvars`-based, auto-added
  to every record.
* `redact(value, keep=4)` / `REDACTED` — mask secrets keeping a short prefix.
* `RedactingFilter` — masks values whose keys match
  `key|token|secret|password|authorization|api` (case-insensitive).
* `truncate_for_log(text, max_chars=200)` — caps long text. When
  `log_document_content` is `False` (default), document-like fields are
  truncated automatically so raw content is never logged.

### Observability hub (`observer.py`)

Implements `rag_core.protocols.Observer`:

* `record(event, attributes=None)` — synchronous; logs at info, increments
  `rag.event.<event>`, and best-effort dispatches listeners on the running
  event loop.
* `emit(event, attributes=None)` — async; same as above **plus** awaits all
  listeners via `asyncio.gather(..., return_exceptions=True)` (listener failures
  are logged, never propagated).
* `subscribe(async_callback)` — register an `async def` listener; returns an
  unsubscribe callable.

```python
from rag_observe import ObservabilityHub

hub = ObservabilityHub()
hub.subscribe(on_event)  # async def on_event(event, attributes) -> None
hub.record("retrieval.started", {"stage": "dense"})
await hub.emit("retrieval.done", {"hits": 7})
```

## Testing

Unit tests run with no OTel SDK installed (everything must work with the no-op
API). Marker: `pytest.mark.unit`.

```
uv run pytest packages/rag-observe -q
```

All calls are safe to keep in production hot paths: with no exporter configured
they collapse to cheap no-ops.
