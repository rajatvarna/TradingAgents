"""Per-run in-memory result cache for data fetching tools.

Eliminates redundant API calls when multiple analysts request the same
(ticker, date, data_type) within a single analysis run.
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = Lock()
_CACHE: dict[tuple, Any] = {}
_HIT_COUNT = 0
_MISS_COUNT = 0


def reset() -> None:
    """Clear the cache. Call this at the start of each analysis run."""
    global _HIT_COUNT, _MISS_COUNT
    with _LOCK:
        _CACHE.clear()
        _HIT_COUNT = 0
        _MISS_COUNT = 0


def stats() -> dict[str, int]:
    """Return hit/miss counts for the current run."""
    with _LOCK:
        return {"hits": _HIT_COUNT, "misses": _MISS_COUNT, "size": len(_CACHE)}


def cached(fn: Callable) -> Callable:
    """Decorator: cache the return value of a data-fetching function by its arguments.

    Only caches successful (non-None, non-empty-string) results so transient
    failures are retried on the next call.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Build a hashable cache key — skip unhashable kwargs gracefully
        try:
            key = (fn.__name__,) + args + tuple(sorted(kwargs.items()))
            hash(key)  # verify hashable
        except TypeError:
            return fn(*args, **kwargs)

        with _LOCK:
            global _HIT_COUNT, _MISS_COUNT
            if key in _CACHE:
                _HIT_COUNT += 1
                logger.debug("run_cache HIT  %s%s", fn.__name__, args[:2])
                return _CACHE[key]

        result = fn(*args, **kwargs)

        if result is not None and result != "":
            with _LOCK:
                _CACHE[key] = result
                _MISS_COUNT += 1
            logger.debug("run_cache MISS %s%s", fn.__name__, args[:2])

        return result

    return wrapper
