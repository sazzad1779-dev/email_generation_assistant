"""
FastAPI route handlers for the Email Generation Assistant.

Each endpoint delegates to the appropriate service layer (LangGraph pipeline,
evaluation engine, comparison runner) and returns standardised Pydantic responses.
"""

from __future__ import annotations

from typing import Any, Dict

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
    ProviderEnum,
    ScenarioResult,
    AggregateScores,
    MetricBreakdown,
    ProviderComparisonResult,
)
from src.config import settings

router = APIRouter(prefix="/v1", tags=["Email Generation"])


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
) -> Dict[str, Any]:
    """Generate an email based on the provided intent, facts, and tone.

    Delegates to the LangGraph pipeline (``src.core.email_generator``).
    The actual pipeline will be connected in Phase 2.
    """
    # ── Placeholder — Phase 2 wires in the real LangGraph pipeline ──
    # TODO: Replace with:
    #   from src.core.email_generator import generate
    #   result = await generate(body)
    #
    # For now return a stub so the API is functional for testing.

    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info(
        "Generate endpoint called",
        intent=body.intent,
        tone=body.tone.value,
        model=body.model.value,
        request_id=request_id,
    )

    placeholder_email = (
        f"**Subject:** {body.intent[:60]}...\n\n"
        f"Dear {body.recipient_name or 'Team'},\n\n"
        f"This email is about: {body.intent}\n\n"
        f"Key points discussed:\n"
        + "\n".join(f"- {fact}" for fact in body.key_facts) +
        "\n\n"
        f"Best regards,\n"
        f"{body.sender_name or '[Your Name]'}"
    )

    return EmailResponse(
        email=placeholder_email,
        metadata={
            "model": body.model.value,
            "tone": body.tone.value,
            "word_count": len(placeholder_email.split()),
            "provider": body.model.value,
            "request_id": request_id,
        },
        quality_flags=["placeholder — LangGraph pipeline not yet connected"],
    ).model_dump()


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
) -> Dict[str, Any]:
    """Evaluate a generated email against the three custom metrics.

    The actual metric computation will be implemented in Phase 6.
    """
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info(
        "Evaluate endpoint called",
        has_reference=body.reference_email is not None,
        request_id=request_id,
    )

    # ── Placeholder — Phase 6 wires in the real metric computation ──
    placeholder = MetricBreakdown(
        score=0.0,
        sub_scores={"placeholder": 0.0},
        explanation="Metrics not yet implemented — Phase 6",
    )

    return EvaluateResponse(
        fact_recall_score=placeholder,
        tone_fidelity_index=placeholder,
        structural_coherence_score=placeholder,
        overall_score=0.0,
        breakdown={"note": "Metric computation pending Phase 6 implementation"},
    ).model_dump()


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
) -> Dict[str, Any]:
    """Run a full batch evaluation across all (or selected) test scenarios.

    The actual batch runner will be implemented in Phase 7.
    """
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info(
        "Batch-evaluate endpoint called",
        model=body.model.value,
        scenarios=body.scenarios,
        request_id=request_id,
    )

    # ── Placeholder — Phase 7 wires in the real evaluation runner ──
    return BatchEvaluateResponse(
        model=body.model,
        scenarios_evaluated=0,
        results=[],
        aggregate=AggregateScores(
            mean_frs=0.0,
            mean_tfi=0.0,
            mean_scs=0.0,
            mean_overall=0.0,
            std_overall=0.0,
            total_scenarios=0,
            failed_scenarios=0,
        ),
        report_path=None,
    ).model_dump()


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
) -> Dict[str, Any]:
    """Run an A/B comparison between two LLM providers.

    The actual comparison logic will be implemented in Phase 7/8.
    """
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info(
        "Compare endpoint called",
        model_a=body.model_a.value,
        model_b=body.model_b.value,
        scenario_count=len(body.scenarios),
        request_id=request_id,
    )

    empty_aggregate = AggregateScores(
        mean_frs=0.0,
        mean_tfi=0.0,
        mean_scs=0.0,
        mean_overall=0.0,
        std_overall=0.0,
        total_scenarios=0,
        failed_scenarios=0,
    )

    # ── Placeholder — Phase 7 wires in the real comparison runner ──
    return CompareResponse(
        model_a_results=ProviderComparisonResult(
            provider=body.model_a,
            aggregate=empty_aggregate,
            results=[],
        ),
        model_b_results=ProviderComparisonResult(
            provider=body.model_b,
            aggregate=empty_aggregate,
            results=[],
        ),
        winner=None,
        margin=0.0,
        failure_analysis={},
        recommendation="Comparison engine not yet implemented — Phase 7",
    ).model_dump()


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
) -> Dict[str, Any]:
    """Check service and provider health.

    Provider status is determined by the availability of API keys and
    the token bucket state. Actual health pings will be added in Phase 4.
    """
    providers: Dict[str, str] = {}

    # Check Gemini
    if settings.GEMINI_API_KEY:
        providers["gemini"] = "healthy"
    else:
        providers["gemini"] = "unconfigured"

    # Check Groq
    if settings.GROQ_API_KEY:
        providers["groq"] = "healthy"
    else:
        providers["groq"] = "unconfigured"

    overall = "healthy" if any(
        s == "healthy" for s in providers.values()
    ) else "degraded"

    return HealthResponse(
        status=overall,
        providers=providers,
        version="1.0.0",
    ).model_dump()
