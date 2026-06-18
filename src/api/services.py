"""
Service layer for the Email Generation Assistant.

Encapsulates business logic so that route handlers remain thin (SRP).
Services depend on abstractions (DIP) and can be unit-tested independently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.api.models import (
    AggregateScores,
    BatchEvaluateResponse,
    CompareRequest,
    CompareResponse,
    EmailRequest,
    EmailResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    MetricBreakdown,
    ProviderComparisonResult,
    ScenarioResult,
)
from src.config import settings
from src.core.rate_limiter import rate_limiter
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmailGenerationService:
    """Handles email generation logic.

    Currently returns placeholder content. In Phase 2+ this will delegate
    to the LangGraph pipeline via an ``EmailGenerator`` abstraction.
    """

    async def generate(self, request: EmailRequest, request_id: str) -> EmailResponse:
        """Generate a single email."""
        logger.info(
            "Generating email",
            intent=request.intent,
            tone=request.tone.value,
            model=request.model.value,
            request_id=request_id,
        )

        # ── Placeholder — Phase 2: replace with LangGraph pipeline ──
        placeholder = (
            f"**Subject:** {request.intent[:60]}...\n\n"
            f"Dear {request.recipient_name or 'Team'},\n\n"
            f"This email is about: {request.intent}\n\n"
            f"Key points discussed:\n"
            + "\n".join(f"- {fact}" for fact in request.key_facts)
            + "\n\n"
            f"Best regards,\n"
            f"{request.sender_name or '[Your Name]'}"
        )

        return EmailResponse(
            email=placeholder,
            metadata={
                "model": request.model.value,
                "tone": request.tone.value,
                "word_count": len(placeholder.split()),
                "provider": request.model.value,
                "request_id": request_id,
            },
            quality_flags=["placeholder — LangGraph pipeline not yet connected"],
        )


class EvaluationService:
    """Handles single and batch email evaluation.

    Currently returns placeholder scores. In Phase 6+ this will run
    the three custom metrics (FRS, TFI, SCS).
    """

    async def evaluate_single(
        self, request: EvaluateRequest, request_id: str
    ) -> EvaluateResponse:
        """Run all three metrics on a single email."""
        logger.info(
            "Evaluating email",
            has_reference=request.reference_email is not None,
            request_id=request_id,
        )

        # ── Placeholder — Phase 6: replace with real metrics ──
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
        )

    async def batch_evaluate(
        self,
        model: str,
        scenarios: Optional[List[int]],
        request_id: str,
    ) -> BatchEvaluateResponse:
        """Run batch evaluation across test scenarios."""
        logger.info(
            "Batch evaluating",
            model=model,
            scenarios=scenarios,
            request_id=request_id,
        )

        # ── Placeholder — Phase 7: replace with real batch runner ──
        return BatchEvaluateResponse(
            model=model,  # type: ignore[arg-type]
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
        )


class ComparisonService:
    """Handles A/B model comparison.

    Currently returns placeholder results. In Phase 7+ this will run
    actual comparison logic.
    """

    async def compare(
        self, request: CompareRequest, request_id: str
    ) -> CompareResponse:
        """Run A/B comparison between two providers."""
        logger.info(
            "Comparing models",
            model_a=request.model_a.value,
            model_b=request.model_b.value,
            scenario_count=len(request.scenarios),
            request_id=request_id,
        )

        empty = AggregateScores(
            mean_frs=0.0,
            mean_tfi=0.0,
            mean_scs=0.0,
            mean_overall=0.0,
            std_overall=0.0,
            total_scenarios=0,
            failed_scenarios=0,
        )

        # ── Placeholder — Phase 7/8: replace with real comparison ──
        return CompareResponse(
            model_a_results=ProviderComparisonResult(
                provider=request.model_a,
                aggregate=empty,
                results=[],
            ),
            model_b_results=ProviderComparisonResult(
                provider=request.model_b,
                aggregate=empty,
                results=[],
            ),
            winner=None,
            margin=0.0,
            failure_analysis={},
            recommendation="Comparison engine not yet implemented — Phase 7",
        )


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

        overall = "healthy" if any(
            s == "healthy" for s in providers.values()
        ) else "degraded"

        return HealthResponse(
            status=overall,
            providers=providers,
            version="1.0.0",
        )
