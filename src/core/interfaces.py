"""
Abstract interfaces for the Email Generation Assistant.

Defines contracts for LLM providers, metrics, and evaluation components
so that high-level modules depend on abstractions (DIP) and new
implementations can be added without modifying existing code (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# ── LLM Provider Interface ──────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""

    content: str
    model_used: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface for LLM providers (Gemini, Groq, etc.)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'gemini', 'groq')."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text from the given prompt."""
        ...

    @abstractmethod
    async def check_health(self) -> bool:
        """Return True if the provider is reachable and operational."""
        ...

    @abstractmethod
    def quota_remaining(self) -> int:
        """Number of requests remaining before hitting the daily limit."""
        ...


# ── Rate Limiter Interface ──────────────────────────────────────────────────


class RateLimiter(ABC):
    """Abstract interface for rate-limit enforcement."""

    @abstractmethod
    async def acquire(self, provider: str, tokens: int = 1) -> float | None:
        """Try to consume *tokens* for *provider*.

        Returns ``None`` if allowed, or ``retry_after`` seconds if denied.
        """
        ...

    @abstractmethod
    def get_status(self, provider: str) -> Dict[str, Any]:
        """Return current rate-limit status for a provider."""
        ...


# ── Metric Interface ────────────────────────────────────────────────────────


class MetricResult(Protocol):
    """Shape of the dict returned by ``Metric.score()``."""
    score: float
    breakdown: Dict[str, Any]


class Metric(ABC):
    """Abstract interface for an evaluation metric."""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        ...

    @property
    @abstractmethod
    def weight(self) -> float:
        """Relative weight when computing an overall score (0–1)."""
        ...

    @abstractmethod
    async def score(self, email: str, **kwargs: Any) -> Dict[str, Any]:
        """Run the metric and return ``{'score': 0-100, 'breakdown': {...}}``."""
        ...


# ── Service Interfaces (for DI / testability) ───────────────────────────────


class EmailGenerator(ABC):
    """Abstract interface for email generation."""

    @abstractmethod
    async def generate(
        self, request: Any, preferred_model: str = "gemini"
    ) -> Any:
        """Generate an email and return an EmailResponse."""
        ...


class EmailEvaluator(ABC):
    """Abstract interface for email evaluation."""

    @abstractmethod
    async def evaluate_single(self, email: str, request: Any) -> Dict[str, Any]:
        """Run all metrics on a single email."""
        ...

    @abstractmethod
    async def batch_evaluate(
        self, scenarios: List[Dict[str, Any]], model: str = "gemini"
    ) -> Any:
        """Run batch evaluation across multiple scenarios."""
        ...
