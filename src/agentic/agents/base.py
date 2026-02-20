"""Base agent setup: Shared LLM configuration and utilities."""

from langchain_ollama import ChatOllama
from config.settings import settings


def get_llm() -> ChatOllama:
    """Get configured Ollama LLM instance.
    
    Uses settings from config for model name and temperature.
    """
    return ChatOllama(
        model=settings.model_name,
        temperature=settings.temperature,
        # base_url=settings.base_url,
        streaming=True,
    )