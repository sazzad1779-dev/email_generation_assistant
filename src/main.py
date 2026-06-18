"""
FastAPI application entry point for the Email Generation Assistant.

Slim factory — delegates middleware, exception handlers, and lifespan
logic to dedicated modules (SRP).  Kept intentionally lean.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.middleware import register_middleware
from src.api.routes import router as api_router
from src.config import settings
from src.utils.logging_config import setup_logging, get_logger

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # ── Startup ──────────────────────────────────────────────────────
    setup_logging()
    logger = get_logger(__name__)

    logger.info(
        "Application starting",
        environment=settings.ENVIRONMENT,
        version="1.0.0",
    )

    try:
        settings.validate_api_keys()
        logger.info("API key validation passed")
    except ValueError:
        logger.warning("No API keys configured — providers unavailable")

    logger.info(
        "Provider status",
        gemini="configured" if settings.GEMINI_API_KEY else "not configured",
        groq="configured" if settings.GROQ_API_KEY else "not configured",
    )

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

# ── Exception Handlers (delegated to dedicated module) ──────────────────────

register_exception_handlers(app)

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
