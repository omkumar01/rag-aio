"""Cache backend implementations.

Each backend satisfies the :class:`rag_core.protocols.Cache` protocol
(``get``/``set``/``delete``) and additionally provides ``clear`` and ``size``.
All I/O-bearing operations are ``async`` and, where they touch the filesystem
or a database, run inside :func:`asyncio.to_thread` so the event loop is never
blocked.
"""

from __future__ import annotations
