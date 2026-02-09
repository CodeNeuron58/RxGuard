"""Logging: Machine-readable, human-friendly, always consistent."""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Call this once at application startup.
    
    Configures both standard library logging and structlog.
    """
    
    # Step 1: Configure standard library (captures logs from dependencies)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Step 2: Configure structlog (your application's structured logs)
    structlog.configure(
        processors=[
            # Filter by level first (performance)
            structlog.stdlib.filter_by_level,
            
            # Add metadata
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            
            # Formatting
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            
            # Output format (JSON for production, pretty for dev)
            structlog.processors.JSONRenderer()
            # Switch to: structlog.dev.ConsoleRenderer(colors=True) for local dev
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> Any:
    """Get a logger bound with the module name.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("event_name", key="value", patient_id="123")
    """
    return structlog.get_logger(name)

# Clinical safety: Always log critical events
def log_clinical_event(
    logger: Any,
    event_type: str,
    patient_context: dict[str, Any] | None = None,
    medication_context: dict[str, Any] | None = None,
    risk_analysis: dict[str, Any] | None = None,
    safety_flag: dict[str, Any] | None = None,
    **kwargs
) -> None:
    """Standardized logging for clinical decisions.
    
    Ensures all safety-critical events have consistent structure.
    Accepts dictionaries corresponding to RxGuard Pydantic models.
    """
    logger.info(
        event_type,
        patient_context=patient_context,
        medication_context=medication_context,
        risk_analysis=risk_analysis,
        safety_flag=safety_flag,
        clinical_event=True,
        **kwargs
    )