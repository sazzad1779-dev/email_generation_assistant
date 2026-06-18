"""
Email generation service.

Encapsulates the logic of producing an email from an ``EmailRequest``.
Currently returns placeholder content; Phase 2+ delegates to the
LangGraph pipeline.
"""

from __future__ import annotations

from src.api.models.requests import EmailRequest
from src.api.models.responses import EmailResponse
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
