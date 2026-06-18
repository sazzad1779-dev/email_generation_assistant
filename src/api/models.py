"""
Pydantic v2 models for request validation and response serialization.

All input models use strict validators to ensure data quality before
reaching the generation or evaluation pipeline.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logging_config import get_logger
from src.utils.validators import (
    check_long_facts,
    sanitize_name,
    validate_facts_not_empty,
    validate_intent_specific,
)

logger = get_logger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────


class ToneEnum(str, Enum):
    """Supported email tones."""

    FORMAL = "formal"
    CASUAL = "casual"
    URGENT = "urgent"
    EMPATHETIC = "empathetic"
    PERSUASIVE = "persuasive"
    APOLOGETIC = "apologetic"


class ProviderEnum(str, Enum):
    """Available LLM providers."""

    GEMINI = "gemini"
    GROQ = "groq"


# ── Request Models ───────────────────────────────────────────────────────────


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


# ── Response Models ──────────────────────────────────────────────────────────


class MetricBreakdown(BaseModel):
    """Detailed sub-scores for a single metric."""

    score: float = Field(..., ge=0, le=100, description="Overall metric score (0–100)")
    sub_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Named sub-component scores",
    )
    explanation: str = Field(
        "",
        description="Human-readable explanation of the score",
    )


class EvaluateResponse(BaseModel):
    """Results of running all three custom metrics on a single email."""

    fact_recall_score: MetricBreakdown = Field(
        ..., description="Fact Recall Score (FRS)"
    )
    tone_fidelity_index: MetricBreakdown = Field(
        ..., description="Tone Fidelity Index (TFI)"
    )
    structural_coherence_score: MetricBreakdown = Field(
        ..., description="Structural Coherence Score (SCS)"
    )
    overall_score: float = Field(
        ..., ge=0, le=100, description="Weighted average of all three metrics"
    )
    breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw per-metric breakdown for debugging",
    )


class EmailResponse(BaseModel):
    """Response from a single email generation request."""

    email: str = Field(..., description="Generated email in Markdown format")
    metadata: Dict[str, Any] = Field(
        ...,
        description="Generation metadata (model, tokens, latency, etc.)",
    )
    quality_flags: List[str] = Field(
        default_factory=list,
        description="Warnings or quality issues detected",
    )


class ScenarioResult(BaseModel):
    """Result of evaluating a single test scenario."""

    scenario_id: int = Field(..., description="Scenario index (1–10)")
    intent: str = Field(..., description="Scenario intent text")
    tone: ToneEnum = Field(..., description="Scenario tone")
    generated_email: str = Field(..., description="Generated email content")
    metrics: EvaluateResponse = Field(..., description="Computed metric scores")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")


class AggregateScores(BaseModel):
    """Aggregated statistics across all evaluated scenarios."""

    mean_frs: float = Field(..., ge=0, le=100, description="Mean Fact Recall Score")
    mean_tfi: float = Field(..., ge=0, le=100, description="Mean Tone Fidelity Index")
    mean_scs: float = Field(..., ge=0, le=100, description="Mean Structural Coherence Score")
    mean_overall: float = Field(..., ge=0, le=100, description="Mean overall score")
    std_overall: float = Field(
        ..., ge=0, description="Standard deviation of overall scores"
    )
    total_scenarios: int = Field(..., description="Number of scenarios evaluated")
    failed_scenarios: int = Field(
        ..., description="Number of scenarios that failed during generation"
    )


class BatchEvaluateResponse(BaseModel):
    """Response from a full batch evaluation run."""

    model: ProviderEnum = Field(..., description="Provider that was evaluated")
    scenarios_evaluated: int = Field(
        ..., description="Number of scenarios successfully evaluated"
    )
    results: List[ScenarioResult] = Field(
        ..., description="Per-scenario results (in order)"
    )
    aggregate: AggregateScores = Field(
        ..., description="Aggregated statistics across all scenarios"
    )
    report_path: Optional[str] = Field(
        None, description="Path to the generated report file"
    )


class ProviderComparisonResult(BaseModel):
    """Results for a single provider in a comparison run."""

    provider: ProviderEnum = Field(..., description="Provider name")
    aggregate: AggregateScores = Field(
        ..., description="Aggregated scores for this provider"
    )
    results: List[ScenarioResult] = Field(
        ..., description="Per-scenario results"
    )


class CompareResponse(BaseModel):
    """Response from an A/B model comparison run."""

    comparison_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique comparison run identifier",
    )
    model_a_results: ProviderComparisonResult = Field(
        ..., description="Results for Model A"
    )
    model_b_results: ProviderComparisonResult = Field(
        ..., description="Results for Model B"
    )
    winner: Optional[ProviderEnum] = Field(
        None,
        description="Provider with higher mean overall score (None if tie)",
    )
    margin: float = Field(
        ..., description="Score margin between winner and loser (absolute)"
    )
    failure_analysis: Dict[str, int] = Field(
        default_factory=dict,
        description="Failure counts per provider",
    )
    recommendation: str = Field(
        ..., description="Recommended provider based on comparison"
    )


class HealthResponse(BaseModel):
    """Response from the health-check endpoint."""

    status: str = Field(..., description="Overall service status", examples=["healthy"])
    providers: Dict[str, str] = Field(
        ...,
        description="Per-provider status (healthy | degraded | unavailable)",
        examples=[{"gemini": "healthy", "groq": "healthy"}],
    )
    version: str = Field(
        ..., description="Application version", examples=["1.0.0"]
    )


class ErrorResponse(BaseModel):
    """Standard error response used by all exception handlers."""

    error_code: str = Field(
        ..., description="Machine-readable error code",
        examples=["rate_limit_exceeded", "llm_failure"],
    )
    message: str = Field(
        ..., description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context"
    )
    request_id: Optional[str] = Field(
        None, description="Request ID for tracing"
    )
    retry_after: Optional[float] = Field(
        None, description="Seconds to wait before retrying (for 429 errors)"
    )
