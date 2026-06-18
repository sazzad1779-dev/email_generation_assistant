"""
A/B comparison service.

Handles model-to-model comparison logic.  Currently returns placeholder
results; Phase 7+ runs actual comparison.
"""

from __future__ import annotations

from src.api.models.requests import CompareRequest
from src.api.models.responses import (
    AggregateScores,
    CompareResponse,
    ProviderComparisonResult,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


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
