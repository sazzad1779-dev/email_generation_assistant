"""
Shared / shorthand models used across multiple endpoints.

``ErrorResponse`` is the standard error body returned by all exception
handlers.  Kept here so both ``responses.py`` and ``exception_handlers.py``
can import it without circular dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response used by all exception handlers."""

    error_code: str = Field(
        ...,
        description="Machine-readable error code",
        examples=["rate_limit_exceeded", "llm_failure"],
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error context",
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing",
    )
    retry_after: Optional[float] = Field(
        None,
        description="Seconds to wait before retrying (for 429 errors)",
    )
