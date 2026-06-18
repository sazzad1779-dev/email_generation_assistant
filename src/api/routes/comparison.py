"""
A/B comparison route — ``POST /v1/compare``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_request_id, rate_limit_dependency
from src.api.models import CompareRequest, CompareResponse, ErrorResponse
from src.api.services import ComparisonService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Email Generation"])

comparison_service = ComparisonService()


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=200,
    summary="Run A/B model comparison",
    description=(
        "Compare two LLM providers across a set of test scenarios. "
        "Returns per-provider aggregate scores, a winner, and a "
        "recommendation based on statistical analysis."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
async def compare_models(
    request: Request,
    body: CompareRequest,
    request_id: str = Depends(get_request_id),
    _=Depends(rate_limit_dependency),
) -> CompareResponse:
    """Run an A/B comparison between two LLM providers."""
    logger.info(
        "Compare endpoint called",
        model_a=body.model_a.value,
        model_b=body.model_b.value,
        scenario_count=len(body.scenarios),
        request_id=request_id,
    )
    return await comparison_service.compare(body, request_id)
