"""
Request-ID middleware.

Attaches a unique ``request_id`` to every request for tracing and
logging correlation.
"""

from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique UUID ``request_id`` to every request.

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
