"""
Structured logging configuration using structlog.

Provides:
- JSON-formatted logs for production
- Colored console output for development
- Automatic request_id injection
- File handler with rotation
- Convenience factory function for module-level loggers
"""

import logging
import logging.handlers  # noqa: F401  (RotatingFileHandler)
import os
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from src.config import settings


def _add_request_id(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Inject request_id into every log entry if present in the context.

    This processor reads from structlog's thread-local context vars so that
    request_id is automatically available across async boundaries.
    """
    from structlog.contextvars import get_merged_contextvars

    ctx = get_merged_contextvars()
    if "request_id" in ctx:
        event_dict["request_id"] = ctx["request_id"]
    return event_dict


def _add_environment(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Tag every log entry with the current environment name."""
    event_dict["environment"] = settings.ENVIRONMENT.lower()
    return event_dict


def setup_logging() -> None:
    """Configure structlog once at application startup.

    Call this exactly once — typically in the FastAPI lifespan handler.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    is_dev = settings.is_development
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # ── Shared processors (applied in every rendering pipeline) ──────
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_environment,
        _add_request_id,
        TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # ── Standard library logging bridge ──────────────────────────────
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── Console handler (colored for dev, JSON for prod) ────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if is_dev:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                ConsoleRenderer(colors=True, sort_keys=False),
            ],
            foreign_pre_chain=shared_processors,
        )
    else:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                JSONRenderer(serializer=lambda o: o, indent=None),
            ],
            foreign_pre_chain=shared_processors,
        )

    console_handler.setFormatter(console_formatter)

    # ── File handler (JSON only, rotated at 10 MB) ───────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            JSONRenderer(indent=None),
        ],
        foreign_pre_chain=shared_processors,
    )

    file_handler.setFormatter(file_formatter)

    # ── Root logger ──────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Remove default handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger for the calling module.

    Usage:
        logger = get_logger(__name__)
        logger.info("event happened", key="value")
    """
    return structlog.get_logger(name or __name__)
