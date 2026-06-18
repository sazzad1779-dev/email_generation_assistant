"""
Health-check service.

Reports overall service health and per-provider status by querying
the rate-limiter state and configuration.
"""

from __future__ import annotations

from typing import Dict

from src.api.models.responses import HealthResponse
from src.config import settings
from src.core.rate_limiter import rate_limiter
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HealthService:
    """Service for health-check logic."""

    async def check_health(self) -> HealthResponse:
        """Check service and provider health."""
        providers: Dict[str, str] = {}

        # Check Gemini
        if settings.GEMINI_API_KEY:
            status = rate_limiter.get_provider_status("gemini")
            providers["gemini"] = "healthy" if status["available"] else "degraded"
        else:
            providers["gemini"] = "unconfigured"

        # Check Groq
        if settings.GROQ_API_KEY:
            status = rate_limiter.get_provider_status("groq")
            providers["groq"] = "healthy" if status["available"] else "degraded"
        else:
            providers["groq"] = "unconfigured"

        overall = (
            "healthy"
            if any(s == "healthy" for s in providers.values())
            else "degraded"
        )

        return HealthResponse(
            status=overall,
            providers=providers,
            version="1.0.0",
        )
