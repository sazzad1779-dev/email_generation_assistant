"""
Domain exception classes for the Email Generation Assistant.

These are pure domain exceptions decoupled from any web framework (DIP).
Exception handlers in the API layer convert them to HTTP responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class EmailGenException(Exception):
    """Base exception for all application-level errors."""


@dataclass
class RateLimitException(EmailGenException):
    """Raised when a rate limit is hit."""

    retry_after: int = 60
    message: str = "Rate limit exceeded. Please wait before retrying."
    provider: Optional[str] = None


@dataclass
class LLMFailureException(EmailGenException):
    """Raised when an LLM provider call fails unexpectedly."""

    provider: str = "unknown"
    message: str = "LLM provider call failed"
    detail: str = ""


@dataclass
class ProviderQuotaExhausted(EmailGenException):
    """Raised when a provider has exhausted its daily quota."""

    provider: str = "unknown"
    reset_at: str = ""
    message: str = "Provider quota exhausted"


@dataclass
class ProviderNotAvailable(EmailGenException):
    """Raised when no provider is currently available."""

    message: str = (
        "No LLM provider is currently available. "
        "Please check your API keys and rate limits, or try again later."
    )


@dataclass
class EmailGenerationError(EmailGenException):
    """Raised when the email generation pipeline fails."""

    message: str = "Email generation failed"
    stage: str = "unknown"
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
