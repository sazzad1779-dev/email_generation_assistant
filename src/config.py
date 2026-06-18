"""
Application configuration via Pydantic Settings.

All environment variables are loaded from .env file (if present) and override
defaults below. Required API keys raise clear errors if missing at runtime.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider API Keys ──────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── Application Settings ───────────────────────────────────────────
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ── Generation Defaults ────────────────────────────────────────────
    MAX_WORDS_DEFAULT: int = 300
    TEMPERATURE: float = 0.0
    TOP_P: float = 0.95

    # ── Provider Rate Limits ───────────────────────────────────────────
    GEMINI_RPM: int = 10
    GEMINI_DAILY_LIMIT: int = 1500
    GROQ_RPM: int = 30
    GROQ_DAILY_LIMIT: int = 1000

    # ── Judge LLM Config ──────────────────────────────────────────────
    JUDGE_MODEL: str = "gemini-2.5-flash"
    JUDGE_TEMPERATURE: float = 0.0
    JUDGE_RUNS: int = 3

    def validate_api_keys(self) -> None:
        """Check that at least one API key is provided.

        Raises a clear ValueError if both keys are empty, guiding the user
        to configure their .env file.
        """
        if not self.GEMINI_API_KEY and not self.GROQ_API_KEY:
            raise ValueError(
                "No API keys configured. "
                "Create a `.env` file in the project root with:\n"
                "  GEMINI_API_KEY=your_google_api_key\n"
                "  GROQ_API_KEY=your_groq_api_key\n"
                "At least one key is required to run the service."
            )

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


# Singleton instance — import `settings` wherever needed.
settings = Settings()
