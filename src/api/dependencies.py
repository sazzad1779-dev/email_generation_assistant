"""
FastAPI dependency injection functions.

Provides reusable dependencies for rate limiting, request context,
and provider checks — all delegating to the core service layer.
"""

from __future__ import annotations

from fastapi import Request
from fastapi import Depends  # noqa: F401 — re-exported for convenience

from src.core.exceptions import RateLimitException
from src.core.rate_limiter import rate_limiter


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

    Delegates to the core ``RateLimiterService`` (sliding-window).

    Attaches ``rate_limit_remaining`` to ``request.state`` for the
    logging middleware to include in response headers.

    Raises ``RateLimitException`` if the client exceeds limits.
    """
    client_id = _get_client_id(request)
    retry_after = rate_limiter.check_client_limit(client_id)

    if retry_after is not None:
        raise RateLimitException(retry_after=int(retry_after))

    request.state.rate_limit_remaining = rate_limiter.client_remaining


async def check_provider_availability(provider: str = "gemini") -> None:
    """Check whether a provider has quota available before processing.

    This is a pre-flight check — the actual consumption happens inside
    the generation pipeline.  If the bucket is empty we reject early
    to avoid wasting time and tokens.
    """
    retry_after = rate_limiter.consume_provider(provider, tokens=0)  # peek
    if retry_after is not None:
        raise RateLimitException(retry_after=int(retry_after))


async def get_request_id(request: Request) -> str:
    """Extract the current request_id from the request state.

    Used as a dependency in endpoint functions that need to log or
    return the request ID.
    """
    return getattr(request.state, "request_id", "no-request-id")
