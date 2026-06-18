"""
Rate limiter with per-provider token buckets and per-client sliding window.

Extracted from the API layer (SRP) so it can be reused by services, CLI
tools, and the API alike.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

from src.config import settings
from src.core.exceptions import RateLimitException


# ── Token Bucket (per-provider) ─────────────────────────────────────────────


class TokenBucket:
    """Token bucket rate limiter for a single provider.

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

        Returns ``retry_after`` seconds if denied, or ``None`` to proceed.
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

        deficit = tokens - self.tokens
        retry_after = deficit / self.refill_rate
        return max(retry_after, 0.1)

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self.tokens

    @property
    def remaining_daily(self) -> int:
        self._check_daily()
        return max(0, self.daily_limit - self.daily_usage)


# ── Rate Limiter Service ────────────────────────────────────────────────────


class RateLimiterService:
    """Central rate-limiter managing buckets for all providers and clients."""

    def __init__(self) -> None:
        # Per-provider token buckets
        self._buckets: Dict[str, TokenBucket] = {
            "gemini": TokenBucket(settings.GEMINI_RPM, settings.GEMINI_DAILY_LIMIT),
            "groq": TokenBucket(settings.GROQ_RPM, settings.GROQ_DAILY_LIMIT),
        }
        # Per-client sliding window
        self._client_window_sec = 60
        self._client_max_requests = 100
        self._client_windows: Dict[str, List[float]] = defaultdict(list)

    # ── Provider token-bucket methods ────────────────────────────────

    def consume_provider(self, provider: str, tokens: int = 1) -> float | None:
        """Try to consume tokens for a provider."""
        bucket = self._buckets.get(provider.lower())
        if bucket is None:
            return None  # Unknown provider — skip rate limiting
        return bucket.consume(tokens)

    def check_provider_available(self, provider: str) -> bool:
        """Peek whether a provider has capacity without consuming."""
        bucket = self._buckets.get(provider.lower())
        if bucket is None:
            return False
        retry_after = bucket.consume(tokens=0)
        return retry_after is None

    def get_provider_status(self, provider: str) -> dict:
        """Return current status for a provider."""
        bucket = self._buckets.get(provider.lower())
        if bucket is None:
            return {"available": False, "tokens": 0, "remaining_daily": 0}
        return {
            "available": bucket.available_tokens > 0,
            "tokens": round(bucket.available_tokens, 1),
            "remaining_daily": bucket.remaining_daily,
        }

    def get_all_provider_statuses(self) -> dict:
        """Return status for all configured providers."""
        return {
            name: self.get_provider_status(name) for name in self._buckets
        }

    # ── Per-client sliding-window methods ────────────────────────────

    def check_client_limit(self, client_id: str) -> float | None:
        """Check if client has exceeded rate limit.

        Returns ``retry_after`` or ``None`` to proceed.
        """
        now = time.monotonic()
        timestamps = self._client_windows[client_id]
        cutoff = now - self._client_window_sec

        # Prune expired entries
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        if len(timestamps) >= self._client_max_requests:
            oldest = timestamps[0]
            retry_after = max(oldest + self._client_window_sec - now, 1.0)
            return retry_after

        timestamps.append(now)
        self._client_windows[client_id] = timestamps
        remaining = max(0, self._client_max_requests - len(timestamps))
        return None  # Proceed

    @property
    def client_remaining(self, client_id: str) -> int:
        timestamps = self._client_windows.get(client_id, [])
        cutoff = time.monotonic() - self._client_window_sec
        active = [t for t in timestamps if t >= cutoff]
        return max(0, self._client_max_requests - len(active))


# Singleton — import `rate_limiter` wherever needed.
rate_limiter = RateLimiterService()
