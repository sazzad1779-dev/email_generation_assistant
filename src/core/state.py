"""
LangGraph state definition for the email generation pipeline.

Defines the ``EmailGenerationState`` TypedDict that flows through all
graph nodes.  Every node reads from and writes back to this dictionary,
making the pipeline fully inspectable and testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from src.api.models.enums import ToneEnum
from src.api.models.requests import EmailRequest


class EmailGenerationState(TypedDict):
    """Complete state that flows through the LangGraph pipeline.

    Attributes
    ----------
    request : EmailRequest
        The original user request — treated as immutable throughout the graph.
    intent_category : str
        Detected category of the intent (e.g. "follow-up", "request",
        "announcement", "escalation", "congratulations", "proposal",
        "apology", "policy", "persuasion", "condolences").
    complexity : str
        Estimated complexity ("low", "medium", "high") based on facts + tone.
    prompt : str
        The fully assembled prompt sent to the LLM.
    raw_output : str
        Raw text returned by the LLM (before any post-processing).
    cleaned_email : str
        Final cleaned and structured email after post-processing.
    metadata : Dict[str, Any]
        Generation metadata — model_used, provider, tokens, latency, etc.
    retry_count : int
        How many times the pipeline has retried (capped at 2).
    error : Optional[str]
        Error message if any node raised an unrecoverable failure.
    scores : Optional[Dict[str, float]]
        Quality-check scores (populated by the ``quality_check`` node).
    quality_passed : bool
        ``True`` if the email passed all heuristic quality gates.
    quality_flags : List[str]
        Human-readable warnings for any quality gates that failed.
    selected_examples : List[Dict[str, Any]]
        Few-shot examples selected for this generation.
    tone_profile : Dict[str, Any]
        Tone-specific guidance loaded from the tone profile registry.
    """

    # Immutable input
    request: EmailRequest

    # Enriched input
    intent_category: str
    complexity: str

    # Prompt assembly
    prompt: str
    selected_examples: List[Dict[str, Any]]
    tone_profile: Dict[str, Any]

    # LLM output
    raw_output: str
    cleaned_email: str
    metadata: Dict[str, Any]

    # Quality / retry
    retry_count: int
    error: Optional[str]
    scores: Optional[Dict[str, float]]
    quality_passed: bool
    quality_flags: List[str]


def create_initial_state(request: EmailRequest) -> EmailGenerationState:
    """Build the initial ``EmailGenerationState`` from a user request.

    All mutable fields get sensible defaults so downstream nodes can
    safely read before writing.
    """
    return {
        "request": request,
        "intent_category": "",
        "complexity": "medium",
        "prompt": "",
        "selected_examples": [],
        "tone_profile": {},
        "raw_output": "",
        "cleaned_email": "",
        "metadata": {},
        "retry_count": 0,
        "error": None,
        "scores": None,
        "quality_passed": False,
        "quality_flags": [],
    }
