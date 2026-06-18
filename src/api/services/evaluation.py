"""
Evaluation service.

Handles single and batch email evaluation.  Currently returns placeholder
scores; Phase 6+ will run the three custom metrics (FRS, TFI, SCS).
"""

from __future__ import annotations

from typing import List, Optional

from src.api.models.requests import EvaluateRequest
from src.api.models.responses import (
    AggregateScores,
    BatchEvaluateResponse,
    EvaluateResponse,
    MetricBreakdown,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


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
