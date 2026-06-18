"""
FastAPI middleware for request tracing, logging, and CORS.

Middleware classes are applied in `src/main.py` during app initialization.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique UUID request_id to every request.

    The request_id is:
    - Added to ``request.state.request_id`` for downstream use
    - Emitted as the ``X-Request-ID`` response header
    - Automatically injected into all structlog log entries for this request
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Make request_id available to structlog context vars
        from structlog.contextvars import bind_contextvars

        bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


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


def setup_cors(app: FastAPI) -> None:
    """Configure CORS for the FastAPI application.

    In development ``ALLOWED_ORIGINS`` defaults to ``["*"]``.  In
    production set the ``ALLOWED_ORIGINS`` env var to a comma-separated
    list (e.g. ``https://app.example.com,https://admin.example.com``).

    Note: ``allow_credentials`` is automatically set to ``False`` when
    ``allow_origins`` is ``["*"]`` to comply with the CORS specification.
    """
    origins = list(settings.ALLOWED_ORIGINS)
    use_wildcard = origins == ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not use_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI application.

    Order matters — the first added middleware runs outermost (last to
    process the request, first to process the response).

    Current order (outer → inner):
      1. CORS
      2. Request ID
      3. Request Logging
    """
    setup_cors(app)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
