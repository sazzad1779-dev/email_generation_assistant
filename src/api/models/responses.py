"""
Pydantic response models for all API endpoints.

Each response model provides rich field metadata for accurate OpenAPI
schema generation and client SDK compatibility.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.api.models.enums import ProviderEnum, ToneEnum


# ── Metric Shared ────────────────────────────────────────────────────────────


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


# ── Generation ───────────────────────────────────────────────────────────────


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


# ── Evaluation ───────────────────────────────────────────────────────────────


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
    std_overall: float = Field(..., ge=0, description="Standard deviation of overall scores")
    total_scenarios: int = Field(..., description="Number of scenarios evaluated")
    failed_scenarios: int = Field(..., description="Number of scenarios that failed during generation")


class BatchEvaluateResponse(BaseModel):
    """Response from a full batch evaluation run."""

    model: ProviderEnum = Field(..., description="Provider that was evaluated")
    scenarios_evaluated: int = Field(..., description="Number of scenarios successfully evaluated")
    results: List[ScenarioResult] = Field(..., description="Per-scenario results (in order)")
    aggregate: AggregateScores = Field(..., description="Aggregated statistics across all scenarios")
    report_path: Optional[str] = Field(None, description="Path to the generated report file")


# ── Comparison ───────────────────────────────────────────────────────────────


class ProviderComparisonResult(BaseModel):
    """Results for a single provider in a comparison run."""

    provider: ProviderEnum = Field(..., description="Provider name")
    aggregate: AggregateScores = Field(..., description="Aggregated scores for this provider")
    results: List[ScenarioResult] = Field(..., description="Per-scenario results")


class CompareResponse(BaseModel):
    """Response from an A/B model comparison run."""

    comparison_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique comparison run identifier",
    )
    model_a_results: ProviderComparisonResult = Field(..., description="Results for Model A")
    model_b_results: ProviderComparisonResult = Field(..., description="Results for Model B")
    winner: Optional[ProviderEnum] = Field(
        None,
        description="Provider with higher mean overall score (None if tie)",
    )
    margin: float = Field(..., description="Score margin between winner and loser (absolute)")
    failure_analysis: Dict[str, int] = Field(
        default_factory=dict,
        description="Failure counts per provider",
    )
    recommendation: str = Field(..., description="Recommended provider based on comparison")


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response from the health-check endpoint."""

    status: str = Field(
        ...,
        description="Overall service status",
        examples=["healthy"],
    )
    providers: Dict[str, str] = Field(
        ...,
        description="Per-provider status (healthy | degraded | unavailable)",
        examples=[{"gemini": "healthy", "groq": "healthy"}],
    )
    version: str = Field(
        ...,
        description="Application version",
        examples=["1.0.0"],
    )
