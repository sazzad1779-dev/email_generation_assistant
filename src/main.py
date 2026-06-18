"""
FastAPI application entry point for the Email Generation Assistant.

Initialises the app, registers middleware, exception handlers, and routers,
and manages the application lifespan (startup / shutdown).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler as fastapi_http_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.middleware import register_middleware
from src.api.routes import router as api_router
from src.config import settings
from src.core.exceptions import (
    LLMFailureException,
    ProviderNotAvailable,
    ProviderQuotaExhausted,
    RateLimitException,
)
from src.api.models import ErrorResponse
from src.utils.logging_config import setup_logging, get_logger

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Startup:
      - Configure structured logging
      - Validate API keys (at least one must be present)
      - Log initial health state

    Shutdown:
      - Clean up any remaining resources
    """
    # ── Startup ──────────────────────────────────────────────────────
    setup_logging()
    logger = get_logger(__name__)

    logger.info(
        "Application starting",
        environment=settings.ENVIRONMENT,
        version="1.0.0",
    )

    # Validate that at least one API key is configured
    try:
        settings.validate_api_keys()
        logger.info("API key validation passed")
    except ValueError as exc:
        logger.warning("No API keys configured — service will start but providers unavailable")

    logger.info(
        "Provider status",
        gemini="configured" if settings.GEMINI_API_KEY else "not configured",
        groq="configured" if settings.GROQ_API_KEY else "not configured",
    )

    # ══ Yield control to the application ══════════════════════════════
    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Application shutting down")


# ── App Factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Generation Assistant",
    description=(
        "An intelligent email generation service powered by LangGraph orchestration "
        "and custom evaluation metrics. Supports multiple LLM providers (Gemini, Groq) "
        "with automatic failover, rate-limit management, and A/B comparison."
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Email Gen Assistant Team",
        "url": "https://github.com/email-generation-assistant",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Email Generation",
            "description": "Generate, evaluate, and compare professional emails",
        },
        {
            "name": "Health",
            "description": "Service health and provider status",
        },
    ],
)

# ── Middleware ────────────────────────────────────────────────────────────────

register_middleware(app)

# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(api_router)

# ── Exception Handlers ───────────────────────────────────────────────────────


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


@app.exception_handler(RateLimitException)
async def rate_limit_handler(request: Request, exc: RateLimitException) -> ErrorResponse:
    """Handle rate limit errors → HTTP 429."""
    logger = get_logger(__name__)
    logger.warning(
        "Rate limit hit",
        retry_after=exc.retry_after,
        request_id=getattr(request.state, "request_id", None),
    )
    error = _build_error_response(
        request=request,
        error_code="rate_limit_exceeded",
        message=exc.detail,
        status_code=429,
        retry_after=float(exc.retry_after),
    )
    return error


@app.exception_handler(LLMFailureException)
async def llm_failure_handler(request: Request, exc: LLMFailureException) -> ErrorResponse:
    """Handle LLM provider failures → HTTP 503."""
    logger = get_logger(__name__)
    logger.error(
        "LLM provider failed",
        provider=exc.provider,
        request_id=getattr(request.state, "request_id", None),
    )
    error = _build_error_response(
        request=request,
        error_code="llm_failure",
        message=exc.detail,
        status_code=503,
        details={"provider": exc.provider},
    )
    return error


@app.exception_handler(ProviderQuotaExhausted)
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
    error = _build_error_response(
        request=request,
        error_code="provider_quota_exhausted",
        message=exc.detail,
        status_code=429,
        details={"provider": exc.provider, "reset_at": exc.reset_at},
    )
    return error


@app.exception_handler(ProviderNotAvailable)
async def provider_not_available_handler(
    request: Request, exc: ProviderNotAvailable
) -> ErrorResponse:
    """Handle unavailable providers → HTTP 503."""
    logger = get_logger(__name__)
    logger.error(
        "No provider available",
        request_id=getattr(request.state, "request_id", None),
    )
    error = _build_error_response(
        request=request,
        error_code="provider_not_available",
        message=exc.detail,
        status_code=503,
    )
    return error


@app.exception_handler(RequestValidationError)
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
    error = _build_error_response(
        request=request,
        error_code="validation_error",
        message="Request validation failed",
        status_code=422,
        details={"errors": exc.errors()},
    )
    return error


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> ErrorResponse:
    """Handle all other HTTP exceptions with a standard ErrorResponse."""
    error = _build_error_response(
        request=request,
        error_code=f"http_{exc.status_code}",
        message=str(exc.detail),
        status_code=exc.status_code,
    )
    return error


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> ErrorResponse:
    """Catch-all handler for unhandled exceptions → HTTP 500."""
    logger = get_logger(__name__)
    logger.exception(
        "Unhandled exception",
        exc_info=exc,
        request_id=getattr(request.state, "request_id", None),
    )
    error = _build_error_response(
        request=request,
        error_code="internal_error",
        message="An unexpected internal error occurred",
        status_code=500,
    )
    return error


# ── Root Endpoint ─────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Redirect to the API documentation."""
    return {
        "service": "Email Generation Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }
