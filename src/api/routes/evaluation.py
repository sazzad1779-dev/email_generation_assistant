"""
Evaluation routes — ``POST /v1/evaluate`` and ``POST /v1/batch-evaluate``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_request_id, rate_limit_dependency
from src.api.models import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
)
from src.api.services import EvaluationService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Email Generation"])

evaluation_service = EvaluationService()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    status_code=200,
    summary="Evaluate a single email",
    description=(
        "Run all three custom metrics (FRS, TFI, SCS) on a generated email "
        "and return detailed scores with sub-score breakdowns."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def evaluate_email(
    request: Request,
    body: EvaluateRequest,
    request_id: str = Depends(get_request_id),
    _=Depends(rate_limit_dependency),
) -> EvaluateResponse:
    """Evaluate a generated email against the three custom metrics."""
    logger.info(
        "Evaluate endpoint called",
        has_reference=body.reference_email is not None,
        request_id=request_id,
    )
    return await evaluation_service.evaluate_single(body, request_id)


@router.post(
    "/batch-evaluate",
    response_model=BatchEvaluateResponse,
    status_code=200,
    summary="Run full batch evaluation",
    description=(
        "Run the complete 10-scenario test suite for a given provider "
        "and return per-scenario metrics plus aggregate statistics."
    ),
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
async def batch_evaluate(
    request: Request,
    body: BatchEvaluateRequest,
    request_id: str = Depends(get_request_id),
    _=Depends(rate_limit_dependency),
) -> BatchEvaluateResponse:
    """Run a full batch evaluation across all (or selected) test scenarios."""
    logger.info(
        "Batch-evaluate endpoint called",
        model=body.model.value,
        scenarios=body.scenarios,
        request_id=request_id,
    )
    return await evaluation_service.batch_evaluate(
        model=body.model.value,
        scenarios=body.scenarios,
        request_id=request_id,
    )
