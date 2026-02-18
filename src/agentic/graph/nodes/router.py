"""Router Node: Manages the cyclic control flow (Logic Hub)."""

from typing import Literal

from src.agentic.state.schemas import RxGuardState
from src.agentic.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)

MAX_ATTEMPTS = 3

def router_node(state: RxGuardState) -> RxGuardState:
    """Routing logic to determine next step based on state.
    
    This node simply increments the attempt counter. 
    Actual routing happens via conditional edges in the graph builder.
    """
    attempts = state.get("attempts", 0) + 1
    state["attempts"] = attempts
    
    logger.info(f"--- ROUTER (Attempt {attempts}/{MAX_ATTEMPTS}) ---")
    return state


def route_after_critic(state: RxGuardState) -> Literal["router", "report"]:
    """Conditional edge logic from Critic."""
    feedback = state.get("critique_feedback")
    
    if feedback and "APPROVE" in feedback:
        return "report"
    
    return "router"


def route_after_router(state: RxGuardState) -> Literal["retrieve", "report"]:
    """Conditional edge logic from Router."""
    attempts = state.get("attempts", 0)
    
    if attempts >= MAX_ATTEMPTS:
        logger.warning("Max attempts reached. Forcing final report.")
        return "report"
        
    return "retrieve"
