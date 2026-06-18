"""
CORS middleware configuration.

Isolated so that CORS rules can be tuned independently of other
middleware logic.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings


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
