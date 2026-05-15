"""Utility functions for logging and retrieval."""

from .logging_config import configure_logging, get_logger, log_clinical_event
from .retrieval import get_embedding_model, get_vectorstore

__all__ = [
    "configure_logging",
    "get_logger",
    "log_clinical_event",
    "get_embedding_model",
    "get_vectorstore",
]
