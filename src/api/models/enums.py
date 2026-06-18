"""
Enum definitions for request/response models.

Isolated in their own module to avoid circular imports — both
``requests.py`` and ``responses.py`` depend on these enums.
"""

from __future__ import annotations

from enum import Enum


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
