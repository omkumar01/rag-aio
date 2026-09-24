# rag-observe

OpenTelemetry **API**-based instrumentation and redacted structured logging for
every rag-aio stage. `rag-core` owns the `Observer` protocol; `rag-observe` is
the default implementation. Depends on `rag-core` and the OpenTelemetry API only --
exporters are optional extras and every span/metric call is a **no-op** when no
SDK is configured.

## Overview

`rag-observe` provides three layers of observability, all designed to be
safe to leave in production hot paths:

1. **Instrumentation** -- context managers (`observe`, `timed`) that wrap pipeline
   stages in OpenTelemetry spans and optional duration histograms. No SDK? No
   overhead beyond a `perf_counter` call.
2. **Structured logging** -- JSON-line formatter, correlation IDs, automatic
   redaction of secret-looking fields, and content truncation that keeps raw
   document text out of logs by default.
3. **Observability hub** -- an `Observer` protocol implementation (`ObservabilityHub`)
   that fans stage events to a structured log, an OTel counter, and a list of
   async listener callbacks.

## Installation

```bash
pip install rag-observe
# or, within the uv workspace:
uv pip install rag-observe
```

For full tracing/metrics/export capabilities, add an exporter extra:

```bash
pip install rag-observe[otlp]      # OTLP/gRPC exporter
pip install rag-observe[console]   # ConsoleSpanExporter for dev
```

```python
from rag_observe import (
    ObservabilityHub,  # Observer protocol implementation
    observe,
    timed,  # span context managers
    StageTimer,  # dependency-free wall-clock timer
    record_counter,
    record_histogram,  # OTel metric helpers (no-op w/o SDK)
    get_logger,  # stdlib logger with JSON output
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

## Architecture / Design Principles

### OpenTelemetry API only, no SDK dependency

`rag-observe` imports `opentelemetry.api` (trace and metrics) -- never the SDK.
`trace.get_tracer("rag-aio")` and `metrics.get_meter("rag-aio")` return no-op
implementations when no SDK is registered. This means:

* Zero configuration required for basic use -- everything works out of the box.
* Full tracing/metrics appear automatically when you add an SDK + exporter.
* Hot-path safety -- no conditional branches on "is tracing enabled?" needed.

### Redaction by default

Secrets are masked mechanically. `RedactingFilter` inspects log record fields and
redacts any value whose key matches `key|token|secret|password|authorization|api`
(case-insensitive). Document content fields (`content|text|body|chunk|document|page`)
are truncated to 200 characters by default via `JSONFormatter` when
`log_document_content=False`.

### Context propagation via `contextvars`

Correlation IDs are stored in a `contextvars.ContextVar`, so they propagate
correctly across `asyncio` tasks within the same task tree. They are included
automatically in every log record emitted by `JSONFormatter`.

### Observer pattern for stage events

`ObservabilityHub` implements `rag_core.protocols.Observer` and exposes both a
synchronous (`record`) and an asynchronous (`emit`) entry point. Listeners are
async callbacks registered via `subscribe`; failures in one listener never break
others or the caller (all listeners run under `asyncio.gather` with
`return_exceptions=True`).

### Design choices

| Concern | Approach | Rationale |
|---|---|---|
| Span creation | `tracer.start_as_current_span` | Preserves OTel context propagation across awaits |
| Duration metric | `timed` records `rag.<stage>.duration_ms` histogram | Separate from span; always emitted even if tracing disabled |
| Metric caching | `_COUNTERS` / `_HISTOGRAMS` dict caches meter instruments | Avoids creating duplicate instruments per call |
| Attribute coercion | `_coerce_attr_value` stringifies non-primitives | Real SDKs reject non-primitive attrs; this normalizes them |

## Source tree

```
src/rag_observe/
  __init__.py          Public API re-exports; version = "0.1.0"
  instrumentation.py   observe(), timed(), record_counter, record_histogram, StageTimer
  logging.py           setup_logging, get_logger, JSONFormatter,
                       RedactingFilter, redact, REDACTED, truncate_for_log,
                       set/get_correlation_id
  observer.py          ObservabilityHub (Observer protocol implementation)
```

## Public API

### Instrumentation (`instrumentation.py`)

#### `observe(stage, attributes=None)`

A synchronous context manager that opens a `rag.<stage>` span, records
`rag.stage` plus any additional `attributes`, measures elapsed time as
`rag.duration_ms`, records exceptions on failure, and re-raises. Fully usable
inside `async def` bodies via `with observe(...): await ...`.

```python
from rag_observe import observe

with observe("retrieval", {"query_id": qid, "strategy": "dense"}):
    hits = await retriever.retrieve(query)
# span "rag.retrieval" closed; rag.duration_ms set; exceptions recorded.
```

#### `timed(stage, attributes=None)`

Like `observe` **plus** a `rag.<stage>.duration_ms` histogram. Both the span
and the metric are emitted even when the body raises:

```python
from rag_observe import timed

with timed("generation"):
    result = await generator.generate(request)
```

#### `record_counter(name, value, attributes=None)` / `record_histogram(...)`

Thin wrappers over `opentelemetry.metrics.get_meter("rag-aio")`. No-op without
an SDK.

```python
from rag_observe import record_counter, record_histogram

record_counter("rag.pipeline.completed", 1, {"kind": "ingest"})
record_histogram("rag.chunk.size_tokens", chunk.token_count, {"chunker": "recursive"})
```

#### `StageTimer`

A minimal, dependency-free wall-clock timer for modules that need timing outside
of a span (e.g. to report durations even when no OTel SDK is configured):

```python
from rag_observe import StageTimer

timer = StageTimer().start()
await do_work()
elapsed_ms = timer.stop()
print(f"took {elapsed_ms:.1f} ms")
```

#### Attribute coercion

Attribute values are automatically coerced to OTel-compatible primitives:

* `None` values are dropped (not recorded).
* Enums are unwrapped via `.value`.
* Lists/tuples are recursively coerced; `None` elements are filtered out.
* Everything else is `str(value)`-ified so real SDKs never reject the call.

```python
with observe(
    "parsing", {"model": "x", "n": 3, "ok": True, "tags": ["a", "b"], "status": SomeEnum.ok}
):
    ...
# None values omitted; enum unwrapped; list coerced.
```

### Structured logging (`logging.py`)

#### `setup_logging(level="INFO", json_output=True, log_document_content=False)`

Configures the root logger idempotently. Previously installed managed handlers
are replaced; foreign handlers are left untouched.

```python
from rag_observe import setup_logging, get_logger

setup_logging(level="INFO", json_output=True, log_document_content=False)
logger = get_logger("my_app.pipeline")

logger.info("ingest.started", extra={"fields": {"document_uri": doc.source_uri}})
logger.warning("retrieval.degraded", extra={"fields": {"hits": 3, "model": "fastembed"}})
```

JSON output format:

```json
{"timestamp": "2026-09-24T12:00:00.000+00:00", "level": "INFO", "logger": "my_app.pipeline",
 "message": "ingest.started", "correlation_id": "corr-123",
 "fields": {"document_uri": "s3://bucket/report.pdf"}}
```

#### `get_logger(name)`

Returns a stdlib `logging.Logger`. Structured JSON output and redaction are
installed by `setup_logging`; without it the logger behaves like a plain stdlib
logger.

#### `set_correlation_id(value)` / `get_correlation_id()`

`contextvars`-based correlation ID, auto-included in every JSON log record:

```python
from rag_observe import set_correlation_id

set_correlation_id("req-abc-123")
# All subsequent log records include "correlation_id": "req-abc-123"
```

#### `redact(value, keep=4)` / `REDACTED`

```python
from rag_observe import redact, REDACTED

mask = redact("sk-abc123secret", keep=4)
# mask == "sk-a" + REDACTED  ->  "sk-aREDACTED"

short = redact("abc", keep=4)
# short == REDACTED  ->  full value masked
```

#### `RedactingFilter`

A `logging.Filter` that masks values whose keys look like secrets. Add it to
any handler (it is auto-added by `setup_logging`):

```python
from rag_observe import RedactingFilter
import logging

handler = logging.StreamHandler()
handler.addFilter(RedactingFilter())
# Keys matching: key|token|secret|password|authorization|api (case-insensitive)
```

#### `truncate_for_log(text, max_chars=200)`

Caps long text to `max_chars` of content plus a `...[truncated]` suffix:

```python
from rag_observe import truncate_for_log

short = truncate_for_log("hello")  # unchanged
long_str = truncate_for_log("a" * 500, max_chars=200)  # 200 chars + "...[truncated]"
```

#### Automatic document content suppression

When `log_document_content=False` (the default), `JSONFormatter` automatically
truncates any field whose key matches `content|text|body|chunk|document|page`
(case-insensitive). This ensures raw document text is never logged by default:

```python
setup_logging(json_output=True)  # log_document_content defaults to False
logger.info("doc.parsed", extra={"fields": {"text": full_page_text, "n_pages": 5}})
# In the JSON record, "text" is truncated to 200 chars
```

### Observability hub (`observer.py`)

`ObservabilityHub` implements `rag_core.protocols.Observer`. It fans each event
to three sinks:

1. A structured log record (info level), with attributes in `fields`.
2. An OpenTelemetry counter `rag.event.<event>` (no-op without an SDK).
3. A list of async listener callbacks.

```python
from rag_observe import ObservabilityHub

hub = ObservabilityHub()


# Register an async listener
async def on_event(event: str, attributes: dict | None) -> None:
    print(f"event: {event}, attrs: {attributes}")


unsubscribe = hub.subscribe(on_event)

# Synchronous entry point (Observer protocol)
hub.record("retrieval.started", {"stage": "dense"})
# -> logs at info, increments rag.event.retrieval.started, dispatches listener async

# Asynchronous entry point
await hub.emit("retrieval.done", {"hits": 7})
# -> same, but listeners are awaited inline via asyncio.gather(return_exceptions=True)

# Unsubscribe
unsubscribe()
```

| Method | Behavior |
|---|---|
| `record(event, attributes)` | Sync; logs + metric, dispatches listeners on background task if loop is running |
| `emit(event, attributes)` | Async; logs + metric, awaits all listeners inline |
| `subscribe(callback)` | Registers an `async def` listener; returns unsubscribe callable |
| `listeners` | Returns a copy of the current listener list |

**Listener isolation:** If a listener raises, it is logged at WARNING level and
the exception is suppressed. Other listeners are unaffected. This holds for
both `record` (best-effort dispatch) and `emit` (gathered with
`return_exceptions=True`).

## Usage Guides

### Beginner: instrumenting a single stage

```python
import asyncio
from rag_observe import observe, timed, setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("pipeline")


async def retrieve(query: str):
    with timed("retrieval", {"query": query[:50]}):
        # ... your retrieval logic ...
        logger.info("retrieval.complete", extra={"fields": {"hits": 5}})
        return [{"chunk_id": "c1", "score": 0.95}]


asyncio.run(retrieve("what is authentication?"))
```

### Beginner: structured logging with correlation IDs

```python
from rag_observe import setup_logging, get_logger, set_correlation_id

setup_logging(level="DEBUG", json_output=True)
logger = get_logger("app")

set_correlation_id("req-12345")
logger.info("started", extra={"fields": {"user_id": "alice"}})
# JSON: {"timestamp": "...", "level": "INFO", "logger": "app",
#        "message": "started", "correlation_id": "req-12345",
#        "fields": {"user_id": "alice"}}
```

### Intermediate: secret redaction in action

```python
from rag_observe import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("app")

# The RedactingFilter runs on every handler and masks secret-looking keys:
logger.info(
    "llm.call",
    extra={
        "fields": {
            "api_key": "sk-abc123def456",  # -> sk-aREDACTED
            "Authorization": "Bearer xyz789",  # -> BeaREDACTED
            "endpoint": "http://localhost:1234",  # kept as-is
        }
    },
)
# The original value "sk-abc123def456" never appears in any log line.
```

### Intermediate: custom event listener

```python
from rag_observe import ObservabilityHub

hub = ObservabilityHub()


async def audit_sink(event: str, attributes: dict | None) -> None:
    """Persist stage events to an audit log for compliance."""
    if event.startswith("retrieval"):
        await persist_audit_event(event, attributes)


unsubscribe = hub.subscribe(audit_sink)

# Now every hub.record / hub.emit will also call audit_sink
hub.record("retrieval.done", {"hits": 3, "query_id": "q1"})
```

### Advanced: wiring up the full OpenTelemetry pipeline

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from rag_observe import setup_logging, get_logger

# 1. Install the OTel SDK + exporter (rag-observe only uses the API)
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 2. Configure structured logging
setup_logging(level="INFO", json_output=True, log_document_content=False)
logger = get_logger("rag.pipeline")

# 3. Now every observe()/timed() call produces real spans + metrics
from rag_observe import observe

with observe("generation", {"model": "qwen3.8-27b"}):
    ...
# Spans flow to your OTLP collector automatically.
```

### Advanced: StageTimer for non-traced contexts

```python
from rag_observe import StageTimer

# Use when you need timing data but don't want span overhead
timer = StageTimer().start()
result = compute_expensive_thing()
elapsed = timer.stop()

# Still record metrics even without tracing:
from rag_observe import record_histogram

record_histogram("rag.custom.op.duration_ms", elapsed, {"op": "compute"})
```

## Configuration

| Setting | Default | Description |
|---|---|---|
| `level` | `"INFO"` | Root logger level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `json_output` | `True` | JSON-line format when `True`; human-readable when `False` |
| `log_document_content` | `False` | When `False`, fields matching `content\|text\|body\|chunk\|document\|page` are truncated to `max_chars` |
| `REDACTED` constant | `"REDACTED"` | Sentinel used by `redact()` |
| `redact(keep)` | `keep=4` | Characters of prefix to preserve before redaction |
| `truncate_for_log(max_chars)` | `200` | Maximum content characters before truncation |

The correlation ID cell (`_correlation_id`) is a `contextvars.ContextVar`, so
each async task tree gets its own value. HTTP frameworks typically set the
correlation ID at request ingress from an incoming `X-Correlation-ID` header.

## Testing

```bash
uv run pytest packages/rag-observe -q
```

All tests use the `unit` marker and run with **no OpenTelemetry SDK installed**.
Every span/metric call must degrade to a no-op without raising:

| File | Focus |
|---|---|
| `tests/test_smoke.py` | Version string, basic import |
| `tests/test_instrumentation.py` | `observe` exception re-raise, attribute coercion, async body support, `timed` on exception, `StageTimer` lifecycle, metric recording without SDK |
| `tests/test_logging.py` | JSON format, correlation ID propagation, `RedactingFilter` key matching, `redact` prefix preservation, `truncate_for_log`, idempotent `setup_logging` |
| `tests/test_observer.py` | `Observer` protocol conformance, zero-listener safety, async listener dispatch, failing listener isolation, subscribe/unsubscribe, deduplication |

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| `rag-core` | local | `Observer` protocol reference |
| `opentelemetry-api` | `>=1.25` | Tracer and meter APIs (no SDK) |

Optional extras (not installed by default):

| Extra | Packages | Purpose |
|---|---|---|
| `otlp` | `opentelemetry-sdk`, `opentelemetry-exporter-otlp` | Full tracing + OTLP/gRPC export |
| `console` | `opentelemetry-sdk` | Console span exporter for local development |
| `prometheus` | `prometheus-client` | Prometheus metric export |

## Cross-Package Relationships

```
rag-core ──► rag-observe    (implements protocols.Observer)
    │                         │
    │                         │ uses protocols.Observer protocol (one-way)
    │                         ▼
    │                    all other packages
    │                         │
    │    every package calls observe()/timed() for instrumentation
    │                         │
    │    every package uses get_logger() / setup_logging() for logging
    │                         │
    │    ObservabilityHub.record/emit fans events to listeners, logs, metrics
    ▼
```

* `rag-observe` depends on `rag-core` (for the `Observer` protocol only).
* Every other `rag-*` package imports from `rag_observe` to instrument its
  stages (`observe`, `timed`), log structured events
  (`setup_logging`, `get_logger`), and emit stage events
  (`ObservabilityHub.record` / `emit`).
* `rag-observe` never imports from any other `rag-*` package.
* When an OTel SDK + exporter is present (typically installed by the application
  entry point `rag-aio` or the orchestrator), spans and metrics flow
  automatically through the same API calls with no code changes.
