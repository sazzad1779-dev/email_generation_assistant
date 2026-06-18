"""
FastAPI route handlers for the Email Generation Assistant.

Thin route layer — each endpoint validates input, delegates to a service,
and returns a response. Business logic lives in ``src.api.services`` (SRP).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import rate_limit_dependency, get_request_id
from src.api.models import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    CompareRequest,
    CompareResponse,
    EmailRequest,
    EmailResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
)
from src.api.services import (
    ComparisonService,
    EmailGenerationService,
    EvaluationService,
    HealthService,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Email Generation"])

# Service instances (can be swapped via DI in tests)
generation_service = EmailGenerationService()
evaluation_service = EvaluationService()
comparison_service = ComparisonService()
health_service = HealthService()


# ── POST /generate ──────────────────────────────────────────────────────────


@router.post(
    "/generate",
    response_model=EmailResponse,
    status_code=200,
    summary="Generate a single email",
    description=(
        "Generate a professional email using the specified LLM provider "
        "(default: Gemini 2.5 Flash). The email is crafted using a "
        "chain-of-thought prompt with dynamic few-shot examples."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
async def generate_email(
    request: Request,
    body: EmailRequest,
    request_id: str = Depends(get_request_id),
    _=Depends(rate_limit_dependency),
) -> EmailResponse:
    """Generate an email based on the provided intent, facts, and tone."""
    logger.info(
        "Generate endpoint called",
        intent=body.intent,
        tone=body.tone.value,
        model=body.model.value,
        request_id=request_id,
    )
    return await generation_service.generate(body, request_id)


# ── POST /evaluate ────────────────────────────────────────────────────────────


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


# ── POST /batch-evaluate ──────────────────────────────────────────────────────


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


# ── POST /compare ─────────────────────────────────────────────────────────────


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


# ── GET /health ───────────────────────────────────────────────────────────────


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
