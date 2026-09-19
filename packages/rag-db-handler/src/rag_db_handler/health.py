"""Aggregate health-check utility for stores."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any


async def check_health(stores: Mapping[str, Any]) -> dict[str, bool]:
    """Probe each named store's ``health()`` coroutine.

    Stores without an async ``health`` callable are assumed healthy (``True``).
    Any exception while probing marks that store unhealthy (``False``) rather
    than aborting the whole probe, so one failing backend never hides the rest.
    """
    results: dict[str, bool] = {}
    for name, store in stores.items():
        health = getattr(store, "health", None)
        if health is None or not callable(health):
            results[name] = True
            continue
        try:
            outcome = health()
            if inspect.isawaitable(outcome):
                outcome = await outcome
            results[name] = bool(outcome)
        except Exception:
            results[name] = False
    return results


__all__ = ["check_health"]
