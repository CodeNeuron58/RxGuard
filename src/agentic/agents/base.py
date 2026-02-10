"""Base agent setup: Shared LLM configuration and utilities."""

from langchain_groq import ChatGroq
from pydantic import SecretStr
from config.settings import settings


def get_llm() -> ChatGroq:
    """Get configured Groq LLM instance.
    
    Uses settings from config for model name, temperature, and max tokens.
    """
    return ChatGroq(
        model=settings.model_name,
        temperature=settings.temperature,
        # max_tokens=settings.max_tokens,
        api_key=SecretStr(settings.GROQ_API_KEY),
    )