"""
FastAPI dependency injection functions.

Provides reusable dependencies for rate limiting, provider availability
checks, and request context extraction.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import Request
from fastapi import Depends  # noqa: F401 — re-exported for convenience

from src.config import settings
from src.core.exceptions import ProviderNotAvailable, RateLimitException


# ── In-Memory Token Bucket (per-provider) ────────────────────────────────


class TokenBucket:
    """Simple token bucket rate limiter.

    Each provider gets its own bucket with RPM-based refill rate and
    a daily cap.
    """

    def __init__(self, rpm: int, daily_limit: int) -> None:
        self.max_tokens = rpm
        self.tokens = float(rpm)
        self.refill_rate = rpm / 60.0  # tokens per second
        self.last_refill = time.monotonic()
        self.daily_limit = daily_limit
        self.daily_usage = 0
        self.daily_reset = time.time() + 86400

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def _check_daily(self) -> None:
        now = time.time()
        if now >= self.daily_reset:
            self.daily_usage = 0
            self.daily_reset = now + 86400

    def consume(self, tokens: int = 1) -> float | None:
        """Try to consume *tokens* from the bucket.

        Returns ``retry_after`` seconds (float) if the request should be
        delayed, or ``None`` if the request can proceed immediately.
        """
        self._refill()
        self._check_daily()

        if self.daily_usage >= self.daily_limit:
            retry_after = self.daily_reset - time.time()
            return max(retry_after, 1.0)

        if self.tokens >= tokens:
            self.tokens -= tokens
            self.daily_usage += tokens
            return None  # Proceed

        # Not enough tokens — compute wait time
        deficit = tokens - self.tokens
        retry_after = deficit / self.refill_rate
        return max(retry_after, 0.1)


# Singleton buckets keyed by provider name
_buckets: dict[str, TokenBucket] = {
    "gemini": TokenBucket(settings.GEMINI_RPM, settings.GEMINI_DAILY_LIMIT),
    "groq": TokenBucket(settings.GROQ_RPM, settings.GROQ_DAILY_LIMIT),
}

# ── Per-Client Sliding-Window Rate Limiter ─────────────────────────────

_CLIENT_WINDOW_SEC = 60       # sliding window width (seconds)
_CLIENT_MAX_REQUESTS = 100    # max requests per window per client
# Map: client_id -> deque of request timestamps (monotonic seconds)
_client_windows: defaultdict[str, list[float]] = defaultdict(list)


def _prune_client_window(client_id: str, now: float) -> list[float]:
    """Remove timestamps older than the sliding window and return the rest."""
    timestamps = _client_windows[client_id]
    cutoff = now - _CLIENT_WINDOW_SEC
    # Pop oldest entries that fall outside the window
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)
    if not timestamps:
        del _client_windows[client_id]
        return []
    return timestamps


def _get_client_id(request: Request) -> str:
    """Extract a client identifier from the request.

    Uses the ``X-Client-ID`` header if present, otherwise falls back to
    the client IP address.
    """
    client_id = request.headers.get("X-Client-ID")
    if client_id:
        return f"header:{client_id}"
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


# ── Dependency Functions ──────────────────────────────────────────────────


async def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces per-client rate limiting.

    Uses a sliding-window algorithm: each client is allowed up to
    ``_CLIENT_MAX_REQUESTS`` requests per 60-second window.  Stale
    entries are pruned on every call, so the ``_client_windows`` dict
    cannot grow unbounded.

    Attaches ``rate_limit_remaining`` to ``request.state`` for the
    logging middleware to include in response headers.

    Raises ``RateLimitException`` (HTTP 429) if the client exceeds limits.
    """
    client_id = _get_client_id(request)
    now = time.monotonic()

    # Prune expired timestamps and get the current count
    timestamps = _prune_client_window(client_id, now)

    if len(timestamps) >= _CLIENT_MAX_REQUESTS:
        # Client is over the limit — oldest timestamp tells us when to retry
        oldest = timestamps[0]
        retry_after = max(oldest + _CLIENT_WINDOW_SEC - now, 1.0)
        raise RateLimitException(retry_after=int(retry_after))

    # Record this request
    timestamps.append(now)
    _client_windows[client_id] = timestamps

    request.state.rate_limit_remaining = max(
        0, _CLIENT_MAX_REQUESTS - len(timestamps)
    )


async def get_provider_bucket(provider: str) -> TokenBucket:
    """Return the token bucket for a given provider name.

    Raises ``ProviderNotAvailable`` if the provider is unknown.
    """
    bucket = _buckets.get(provider.lower())
    if bucket is None:
        raise ProviderNotAvailable()
    return bucket


async def check_provider_availability(provider: str = "gemini") -> None:
    """Check whether a provider has quota available before processing.

    This is a pre-flight check — the actual consumption happens inside
    the generation pipeline.  If the bucket is empty we reject early
    to avoid wasting time and tokens.
    """
    bucket = _buckets.get(provider.lower())
    if bucket is None:
        raise ProviderNotAvailable()

    retry_after = bucket.consume(tokens=0)  # peek without consuming
    if retry_after is not None:
        raise RateLimitException(retry_after=int(retry_after))

    bucket.consume(tokens=1)


async def get_request_id(request: Request) -> str:
    """Extract the current request_id from the request state.

    Used as a dependency in endpoint functions that need to log or
    return the request ID.
    """
    return getattr(request.state, "request_id", "no-request-id")
