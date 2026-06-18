"""
Request-logging middleware.

Logs every request with method, path, status code, and duration using
structured logging (structlog).
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and duration.

    Uses structlog for structured logging — output is coloured in development
    and JSON in production.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""

        start_time = time.monotonic()
        response: Response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000

        status_code = response.status_code

        log_kwargs = {
            "method": method,
            "path": f"{path}?{query}" if query else path,
            "status": status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if status_code >= 500:
            logger.error("Request failed", **log_kwargs)
        elif status_code >= 400:
            logger.warning("Request warning", **log_kwargs)
        else:
            logger.info("Request completed", **log_kwargs)

        # Add rate-limit headers if present in response
        if hasattr(request.state, "rate_limit_remaining"):
            response.headers["X-RateLimit-Remaining"] = str(
                request.state.rate_limit_remaining
            )

        return response
