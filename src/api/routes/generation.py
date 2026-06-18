"""
Email generation route — ``POST /v1/generate``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_request_id, rate_limit_dependency
from src.api.models import EmailRequest, EmailResponse, ErrorResponse
from src.api.services import EmailGenerationService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Email Generation"])

# Service instance (can be swapped via DI in tests)
generation_service = EmailGenerationService()


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
