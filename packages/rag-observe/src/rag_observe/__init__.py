"""rag-observe: OpenTelemetry instrumentation and redacted structured logging.

Depends on the OpenTelemetry **API** only (no SDK required); when no SDK is
configured every span/metric call is a no-op.
"""

from __future__ import annotations

from .instrumentation import (
    StageTimer,
    observe,
    record_counter,
    record_histogram,
    timed,
)
from .logging import (
    REDACTED,
    JSONFormatter,
    RedactingFilter,
    get_correlation_id,
    get_logger,
    redact,
    set_correlation_id,
    setup_logging,
    truncate_for_log,
)
from .observer import ObservabilityHub

__version__ = "0.1.1"

__all__ = [
    "REDACTED",
    "JSONFormatter",
    "ObservabilityHub",
    "RedactingFilter",
    "StageTimer",
    "get_correlation_id",
    "get_logger",
    "observe",
    "record_counter",
    "record_histogram",
    "redact",
    "set_correlation_id",
    "setup_logging",
    "timed",
    "truncate_for_log",
]
