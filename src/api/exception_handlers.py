"""
Exception handlers for the FastAPI application.

All exception handlers convert domain exceptions into standardised
``ErrorResponse`` models with proper HTTP status codes.

Extracted from main.py to follow Single Responsibility Principle (SRP).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.models import ErrorResponse
from src.core.exceptions import (
    EmailGenerationError,
    LLMFailureException,
    ProviderNotAvailable,
    ProviderQuotaExhausted,
    RateLimitException,
)
from src.utils.logging_config import get_logger


def _build_error_response(
    request: Request,
    error_code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
    retry_after: float | None = None,
) -> ErrorResponse:
    """Build a standardised error response with request_id."""
    request_id = getattr(request.state, "request_id", None)
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        retry_after=retry_after,
    )


async def rate_limit_handler(request: Request, exc: RateLimitException) -> ErrorResponse:
    """Handle rate limit errors → HTTP 429."""
    logger = get_logger(__name__)
    logger.warning(
        "Rate limit hit",
        retry_after=exc.retry_after,
        provider=exc.provider,
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="rate_limit_exceeded",
        message=exc.message,
        status_code=429,
        retry_after=float(exc.retry_after),
    )


async def llm_failure_handler(request: Request, exc: LLMFailureException) -> ErrorResponse:
    """Handle LLM provider failures → HTTP 503."""
    logger = get_logger(__name__)
    logger.error(
        "LLM provider failed",
        provider=exc.provider,
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="llm_failure",
        message=f"LLM provider '{exc.provider}' failed: {exc.detail or exc.message}",
        status_code=503,
        details={"provider": exc.provider},
    )


async def quota_exhausted_handler(
    request: Request, exc: ProviderQuotaExhausted
) -> ErrorResponse:
    """Handle provider quota exhaustion → HTTP 429."""
    logger = get_logger(__name__)
    logger.warning(
        "Provider quota exhausted",
        provider=exc.provider,
        reset_at=exc.reset_at,
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="provider_quota_exhausted",
        message=f"Provider '{exc.provider}' quota exhausted until {exc.reset_at}",
        status_code=429,
        details={"provider": exc.provider, "reset_at": exc.reset_at},
    )


async def provider_not_available_handler(
    request: Request, exc: ProviderNotAvailable
) -> ErrorResponse:
    """Handle unavailable providers → HTTP 503."""
    logger = get_logger(__name__)
    logger.error(
        "No provider available",
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="provider_not_available",
        message=exc.message,
        status_code=503,
    )


async def email_generation_error_handler(
    request: Request, exc: EmailGenerationError
) -> ErrorResponse:
    """Handle email generation failures → HTTP 500."""
    logger = get_logger(__name__)
    logger.error(
        "Email generation failed",
        stage=exc.stage,
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="email_generation_error",
        message=f"Email generation failed at stage '{exc.stage}': {exc.message}",
        status_code=500,
        details={"stage": exc.stage, **exc.metadata},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> ErrorResponse:
    """Handle Pydantic validation errors → HTTP 422."""
    logger = get_logger(__name__)
    logger.warning(
        "Request validation failed",
        errors=exc.errors(),
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="validation_error",
        message="Request validation failed",
        status_code=422,
        details={"errors": exc.errors()},
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> ErrorResponse:
    """Handle all other HTTP exceptions with a standard ErrorResponse."""
    return _build_error_response(
        request=request,
        error_code=f"http_{exc.status_code}",
        message=str(exc.detail),
        status_code=exc.status_code,
    )


async def global_exception_handler(request: Request, exc: Exception) -> ErrorResponse:
    """Catch-all handler for unhandled exceptions → HTTP 500."""
    logger = get_logger(__name__)
    logger.exception(
        "Unhandled exception",
        exc_info=exc,
        request_id=getattr(request.state, "request_id", None),
    )
    return _build_error_response(
        request=request,
        error_code="internal_error",
        message="An unexpected internal error occurred",
        status_code=500,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI application."""
    app.add_exception_handler(RateLimitException, rate_limit_handler)
    app.add_exception_handler(LLMFailureException, llm_failure_handler)
    app.add_exception_handler(ProviderQuotaExhausted, quota_exhausted_handler)
    app.add_exception_handler(ProviderNotAvailable, provider_not_available_handler)
    app.add_exception_handler(EmailGenerationError, email_generation_error_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
