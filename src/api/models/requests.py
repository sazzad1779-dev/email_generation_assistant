"""
Pydantic request models for all API endpoints.

Validation logic is delegated to ``src.utils.validators`` so that
the same rules can be reused by CLI tools, test scripts, or future
interfaces (Interface Segregation Principle).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.models.enums import ProviderEnum, ToneEnum
from src.utils.logging_config import get_logger
from src.utils.validators import (
    check_long_facts,
    sanitize_name,
    validate_facts_not_empty,
    validate_intent_specific,
)

logger = get_logger(__name__)


# ── Email Generation ─────────────────────────────────────────────────────────


class EmailRequest(BaseModel):
    """Single email generation request.

    Validates intent specificity, fact completeness, and tone assignment
    before the request reaches the LangGraph pipeline.
    """

    intent: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Core purpose of the email — be specific (e.g., 'Follow up on Q3 budget approval meeting')",
        examples=["Request a deadline extension for the Q4 marketing report submission"],
    )
    key_facts: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Key facts to include in the email (1–10 items, each ≥5 characters)",
        examples=[["Project deadline is Dec 15", "Client requested additional deliverables"]],
    )
    tone: ToneEnum = Field(
        ...,
        description="Desired tone for the generated email",
        examples=["formal"],
    )
    recipient_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Recipient's name (optional)",
        examples=["Dr. Sarah Chen"],
    )
    sender_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Sender's name (optional — defaults to a professional signature)",
        examples=["John Smith"],
    )
    max_words: int = Field(
        300,
        ge=50,
        le=800,
        description="Target word count for the generated email (50–800)",
        examples=[250],
    )
    model: ProviderEnum = Field(
        ProviderEnum.GEMINI,
        description="LLM provider to use for generation",
    )

    @field_validator("key_facts")
    @classmethod
    def validate_facts_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure each fact is substantive (≥5 chars after stripping)."""
        return validate_facts_not_empty(v)

    @field_validator("intent")
    @classmethod
    def validate_intent_specific(cls, v: str) -> str:
        """Reject vague intents that contain only generic email words."""
        return validate_intent_specific(v)

    @field_validator("recipient_name", "sender_name")
    @classmethod
    def sanitize_name(cls, v: Optional[str]) -> Optional[str]:
        """Strip special characters from names to prevent injection."""
        return sanitize_name(v)

    @model_validator(mode="after")
    def warn_on_long_facts(self) -> "EmailRequest":
        """Log a warning if any fact exceeds 500 characters."""
        check_long_facts(self.key_facts)
        return self


# ── Evaluation ───────────────────────────────────────────────────────────────


class EvaluateRequest(BaseModel):
    """Request to evaluate a single generated email against metrics."""

    email: str = Field(
        ...,
        min_length=20,
        max_length=10_000,
        description="The generated email content to evaluate",
    )
    reference_email: Optional[str] = Field(
        None,
        max_length=10_000,
        description="Optional gold-standard reference email for comparison",
    )
    request: EmailRequest = Field(
        ...,
        description="The original EmailRequest that produced this email",
    )


class BatchEvaluateRequest(BaseModel):
    """Request to run a full batch evaluation across test scenarios."""

    model: ProviderEnum = Field(
        ProviderEnum.GEMINI,
        description="Provider to evaluate",
    )
    scenarios: Optional[List[int]] = Field(
        None,
        description="Specific scenario indices to evaluate (None = all 10)",
        examples=[None, [1, 3, 5]],
    )


# ── Comparison ───────────────────────────────────────────────────────────────


class CompareRequest(BaseModel):
    """Request to run an A/B comparison between two providers."""

    model_a: ProviderEnum = Field(
        ProviderEnum.GEMINI,
        description="First provider (default: Gemini)",
    )
    model_b: ProviderEnum = Field(
        ProviderEnum.GROQ,
        description="Second provider (default: Groq)",
    )
    scenarios: List[int] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Scenario indices to compare (1–10)",
        examples=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
    )
