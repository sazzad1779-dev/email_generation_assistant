"""
Email generation service.

Encapsulates the logic of producing an email from an ``EmailRequest``.
Delegates to the LangGraph pipeline (Phase 2) via the orchestrator in
``src.core.email_generator``.
"""

from __future__ import annotations

from src.api.models.requests import EmailRequest
from src.api.models.responses import EmailResponse
from src.core.email_generator import generate_email as langgraph_generate
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmailGenerationService:
    """Handles email generation logic.

    Delegates to the LangGraph state machine pipeline for actual
    generation.  This service layer adds request tracking and
    any pre/post-processing not handled by the graph.
    """

    async def generate(self, request: EmailRequest, request_id: str) -> EmailResponse:
        """Generate a single email using the LangGraph pipeline."""
        logger.info(
            "Generating email via LangGraph pipeline",
            intent=request.intent[:80],
            tone=request.tone.value,
            model=request.model.value,
            request_id=request_id,
        )

        # Delegate to the LangGraph orchestrator
        response = await langgraph_generate(
            request=request,
            preferred_model=request.model.value,
        )

        # Inject request_id into response metadata
        response.metadata["request_id"] = request_id

        logger.info(
            "Email generated successfully",
            word_count=response.metadata.get("word_count", 0),
            quality_passed=response.metadata.get("quality_passed", False),
            retry_count=response.metadata.get("retry_count", 0),
            request_id=request_id,
        )

        return response
