"""Configuration: Environment-aware, validated, typed.

Rule: If it might change between environments, it belongs here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """All configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra env vars without crashing
    )
    
    # === LLM CONFIGURATION ===
    GROQ_API_KEY: str = Field(description="Groq API key")
    model_name: str = "openai/gpt-oss-120b"
    temperature: float = 0.1  # Low for clinical precision
    max_tokens: int = 2000
    
    # # === CLINICAL SAFETY ===
    # max_iterations: int = 3  # Hard limit on agent loops
    # require_medication_rationale: bool = True  # Must explain why
    # contraindication_check_enabled: bool = True  # Safety feature
    
    # # === OBSERVABILITY ===
    # log_level: str = "INFO"  # DEBUG for troubleshooting
    # langsmith_api_key: str | None = None
    # langsmith_project: str = "clinical-agent-dev"
    
    # # === ENVIRONMENT ===
    # environment: str = "development"  # development, staging, production


@lru_cache()
def get_settings() -> Settings:
    """Singleton pattern — config loaded once, reused everywhere.
    
    Why cache? Loading from disk/env is I/O. Do it once.
    """
    return Settings()


# Export for easy importing: from config.settings import settings
settings = get_settings()