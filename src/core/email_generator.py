"""
Email generation orchestrator using the LangGraph pipeline.

Wraps the compiled LangGraph state machine with a clean ``async generate()``
interface that the service layer calls.  Handles state initialisation,
graph invocation, error mapping, and response assembly.
"""

from __future__ import annotations

from typing import List

from langgraph.graph.state import CompiledStateGraph

from src.api.models.requests import EmailRequest
from src.api.models.responses import EmailResponse
from src.core.exceptions import (
    EmailGenerationError,
    LLMFailureException,
    ProviderNotAvailable,
)
from src.core.graph import build_email_graph
from src.core.state import EmailGenerationState, create_initial_state
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Compiled graph singleton ────────────────────────────────────────────────

_graph: CompiledStateGraph | None = None


def _get_graph() -> CompiledStateGraph:
    """Return the compiled LangGraph, building it once (lazy singleton)."""
    global _graph
    if _graph is None:
        _graph = build_email_graph()
        logger.info("LangGraph pipeline compiled and cached")
    return _graph


# ── Orchestrator ────────────────────────────────────────────────────────────


async def generate_email(
    request: EmailRequest,
    preferred_model: str = "gemini",
) -> EmailResponse:
    """Generate a professional email using the LangGraph pipeline.

    Parameters
    ----------
    request : EmailRequest
        The validated user request containing intent, facts, tone, etc.
    preferred_model : str
        Preferred LLM provider (``"gemini"`` or ``"groq"``).
        Currently used as metadata; Phase 4 will use it for provider selection.

    Returns
    -------
    EmailResponse
        The generated email with metadata and quality flags.

    Raises
    ------
    EmailGenerationError
        If the pipeline encounters an unrecoverable error.
    LLMFailureException
        If all LLM providers fail during generation.
    ProviderNotAvailable
        If no provider has available quota.
    """
    logger.info(
        "Starting email generation pipeline",
        intent=request.intent[:60],
        tone=request.tone.value,
        model=preferred_model,
    )

    # 1. Initialise state
    initial_state = create_initial_state(request)

    # 2. Get the compiled graph
    graph = _get_graph()

    # 3. Run the pipeline
    try:
        final_state: EmailGenerationState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("LangGraph pipeline threw an exception", error=str(exc))
        raise EmailGenerationError(
            stage="pipeline_execution",
            detail=str(exc),
        ) from exc

    # 4. Check for errors in the final state
    if final_state.get("error"):
        error_msg = final_state["error"]
        logger.error("Pipeline completed with error", error=error_msg)

        if "provider" in str(error_msg).lower() or "llm" in str(error_msg).lower():
            raise LLMFailureException(
                provider=preferred_model,
                detail=error_msg,
            )
        raise EmailGenerationError(
            stage="pipeline",
            detail=error_msg,
        )

    # 5. Extract the cleaned email
    cleaned_email = final_state.get("cleaned_email", "")
    if not cleaned_email:
        logger.error("Pipeline produced empty email")
        raise EmailGenerationError(
            stage="post_process",
            detail="Pipeline produced an empty email after processing",
        )

    # 6. Assemble metadata
    metadata = dict(final_state.get("metadata", {}))
    metadata.update({
        "intent_category": final_state.get("intent_category", ""),
        "complexity": final_state.get("complexity", ""),
        "retry_count": final_state.get("retry_count", 0),
        "quality_passed": final_state.get("quality_passed", False),
        "quality_score": (
            final_state.get("scores", {}).get("overall", 0)
            if final_state.get("scores")
            else 0
        ),
    })

    # 7. Assemble quality flags
    quality_flags: List[str] = list(final_state.get("quality_flags", []))
    if not final_state.get("quality_passed", False):
        quality_flags.append("Email did not pass all quality checks — review recommended")

    word_count = len(cleaned_email.split())
    metadata["word_count"] = word_count

    logger.info(
        "Email generation pipeline completed",
        word_count=word_count,
        quality_passed=final_state.get("quality_passed", False),
        retry_count=final_state.get("retry_count", 0),
    )

    return EmailResponse(
        email=cleaned_email,
        metadata=metadata,
        quality_flags=quality_flags,
    )
