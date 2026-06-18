"""
Custom exception classes for the Email Generation Assistant.

All custom exceptions extend FastAPI's HTTPException to ensure
consistent error responses across the application.
"""

from fastapi import HTTPException


class RateLimitException(HTTPException):
    """Raised when a provider rate limit is hit.

    Returns HTTP 429 with a retry_after header hint.
    """

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded. Please wait before retrying.",
        )
        self.retry_after = retry_after


class LLMFailureException(HTTPException):
    """Raised when an LLM provider call fails unexpectedly.

    Returns HTTP 503 with the provider name and error detail.
    """

    def __init__(self, provider: str, detail: str) -> None:
        super().__init__(
            status_code=503,
            detail=f"LLM provider '{provider}' failed: {detail}",
        )
        self.provider = provider


class ProviderQuotaExhausted(HTTPException):
    """Raised when a provider has exhausted its daily quota.

    Returns HTTP 429 with the provider name and reset time.
    """

    def __init__(self, provider: str, reset_at: str) -> None:
        super().__init__(
            status_code=429,
            detail=f"Provider '{provider}' quota exhausted until {reset_at}",
        )
        self.provider = provider
        self.reset_at = reset_at


class ProviderNotAvailable(HTTPException):
    """Raised when no provider is currently available.

    Returns HTTP 503 with suggestions for the user.
    """

    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            detail=(
                "No LLM provider is currently available. "
                "Please check your API keys and rate limits, or try again later."
            ),
        )


class EmailGenerationError(HTTPException):
    """Raised when the email generation pipeline fails.

    Returns HTTP 500 with context about the failure.
    """

    def __init__(self, detail: str, stage: str = "unknown") -> None:
        super().__init__(
            status_code=500,
            detail=f"Email generation failed at stage '{stage}': {detail}",
        )
        self.stage = stage
