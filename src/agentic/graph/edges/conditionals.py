"""Conditional edges: Routing logic for graph decisions."""

from src.agentic.state.schemas import RxGuardState
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)


def confidence_gate(state: RxGuardState) -> str:
    """Route based on extraction confidence.
    
    If confidence is too low, stop the pipeline.
    If confidence is sufficient, continue to retrieval.
    
    Args:
        state: Current graph state with confidence score
        
    Returns:
        "retrieve" to continue, "stop" to end
    """
    confidence = state.get("confidence")
    
    if confidence is None or confidence < 0.75:
        logger.warning(
            "Low confidence extraction, stopping pipeline",
            confidence=confidence,
            threshold=0.75
        )
        return "stop"
    
    logger.info(
        "Confidence sufficient, continuing to retrieval",
        confidence=confidence
    )
    return "retrieve"