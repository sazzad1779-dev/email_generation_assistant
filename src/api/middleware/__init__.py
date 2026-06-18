"""
Middleware registration and re-exports.

``register_middleware()`` is the single entry point called from
``src/main.py``.  Individual middleware classes can also be imported
directly for unit testing.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.middleware.cors import setup_cors
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.request_id import RequestIDMiddleware


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


__all__ = [
    "register_middleware",
    "setup_cors",
    "RequestIDMiddleware",
    "RequestLoggingMiddleware",
]
