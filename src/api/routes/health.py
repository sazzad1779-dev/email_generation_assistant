"""
Health-check route — ``GET /v1/health``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_request_id
from src.api.models import HealthResponse
from src.api.services import HealthService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Health"])

health_service = HealthService()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Service health check",
    description=(
        "Returns the current health status of the service and all "
        "configured LLM providers. Useful for monitoring and load "
        "balancer health probes."
    ),
)
async def health_check(
    request: Request,
    request_id: str = Depends(get_request_id),
) -> HealthResponse:
    """Check service and provider health."""
    logger.info("Health check called", request_id=request_id)
    return await health_service.check_health()
