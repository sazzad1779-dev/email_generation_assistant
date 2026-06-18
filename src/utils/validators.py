"""
Shared validation utilities for the Email Generation Assistant.

Keeps complex validation logic out of Pydantic models (ISP) so that
validators can be reused across different contexts (API, CLI, tests).
"""

from __future__ import annotations

from typing import List, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def validate_facts_not_empty(facts: List[str]) -> List[str]:
    """Ensure each fact is substantive (≥5 chars after stripping)."""
    for i, fact in enumerate(facts):
        stripped = fact.strip()
        if len(stripped) < 5:
            raise ValueError(
                f"Fact #{i + 1} is too short ({len(stripped)} chars). "
                "Each fact must be at least 5 characters."
            )
    return [f.strip() for f in facts]


def validate_intent_specific(intent: str) -> str:
    """Reject vague intents that contain only generic email words."""
    stripped = intent.strip()
    forbidden = {"email", "write", "compose", "draft"}
    words = set(stripped.lower().split())
    if forbidden & words and len(stripped) < 30:
        raise ValueError(
            "Intent is too vague. Be specific — describe the purpose, "
            "e.g., 'Request a deadline extension for Q4 marketing report' "
            "instead of 'write an email'."
        )
    return stripped


def sanitize_name(name: Optional[str]) -> Optional[str]:
    """Strip special characters from names to prevent injection."""
    if name is None:
        return name
    sanitized = "".join(c for c in name if c.isprintable() and c not in "<>\"'&")
    return sanitized.strip() or None


def check_long_facts(facts: List[str]) -> None:
    """Log a warning if any fact exceeds 500 characters."""
    for i, fact in enumerate(facts):
        if len(fact) > 500:
            logger.warning("Fact #%d exceeds 500 chars — may be truncated", i + 1)
