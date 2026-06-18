"""
Re-exports all service classes so consumers can import from
``src.api.services`` without knowing the internal subpackage layout.
"""

from src.api.services.comparison import ComparisonService
from src.api.services.evaluation import EvaluationService
from src.api.services.generation import EmailGenerationService
from src.api.services.health import HealthService

__all__ = [
    "EmailGenerationService",
    "EvaluationService",
    "ComparisonService",
    "HealthService",
]
