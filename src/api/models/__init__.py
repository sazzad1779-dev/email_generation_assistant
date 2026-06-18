"""
Re-exports all Pydantic models so consumers can import from
``src.api.models`` without knowing the internal subpackage layout.
"""

from src.api.models.common import ErrorResponse
from src.api.models.enums import ProviderEnum, ToneEnum
from src.api.models.requests import (
    BatchEvaluateRequest,
    CompareRequest,
    EmailRequest,
    EvaluateRequest,
)
from src.api.models.responses import (
    AggregateScores,
    BatchEvaluateResponse,
    CompareResponse,
    EmailResponse,
    EvaluateResponse,
    HealthResponse,
    MetricBreakdown,
    ProviderComparisonResult,
    ScenarioResult,
)

__all__ = [
    # Enums
    "ToneEnum",
    "ProviderEnum",
    # Common
    "ErrorResponse",
    # Requests
    "EmailRequest",
    "EvaluateRequest",
    "BatchEvaluateRequest",
    "CompareRequest",
    # Responses
    "EmailResponse",
    "EvaluateResponse",
    "BatchEvaluateResponse",
    "CompareResponse",
    "HealthResponse",
    "ScenarioResult",
    "ProviderComparisonResult",
    "AggregateScores",
    "MetricBreakdown",
]
