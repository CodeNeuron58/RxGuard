
"""Graph Builder: Compiles LangGraph with all nodes and edges."""

from langgraph.graph import StateGraph, START, END

from src.agentic.graph.edges import confidence_gate
from src.agentic.graph.nodes import (
    extract_patient_profile,
    planner_node,
    research_agent_node,
    risk_reasoning_node,
    safety_critic_node,
    final_report_node,
    router_node,
    route_after_router,
    route_after_critic
)
from src.agentic.state import RxGuardState
from src.agentic.utils import get_logger

logger = get_logger(__name__)


def route_research_loop(state: RxGuardState):
    """Loop between Planner and Research Agent until all steps are done."""
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)
    
    if current_index < len(plan):
        return "research"
    else:
        return "reason"

def build_graph() -> StateGraph:
    """Build and compile the RxGuard clinical agent graph.
    
    Returns:
        Compiled LangGraph application
    """
    logger.info("Building RxGuard REFLEXIVE graph...")
    
    # Initialize graph with state schema
    graph = StateGraph(RxGuardState)
    
    # Add nodes
    graph.add_node("extract", extract_patient_profile)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("reason", risk_reasoning_node)
    graph.add_node("critic", safety_critic_node)
    graph.add_node("report", final_report_node)
    
    # Add edges
    graph.add_edge(START, "extract")
    
    # Conditional edge: confidence check
    # If high confidence -> Start the Loop (via Router to init attempts)
    graph.add_conditional_edges(
        "extract",
        confidence_gate,
        {
            "retrieve": "router",  # Start the loop
            "stop": END
        }
    )
    
    # Router Logic (Check attempts)
    # Router -> Planner (Start of a new attempt/cycle)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "planner", # CHANGED: Router now points to Planner
            "report": "report"
        }
    )
    
    # Planner -> Research (Start)
    graph.add_edge("planner", "research")
    
    # Research -> Research (Loop) OR Reason (Done)
    graph.add_conditional_edges(
        "research",
        route_research_loop,
        {
            "research": "research", # Next step
            "reason": "reason"      # Done
        }
    )
    
    # Reason -> Critic
    graph.add_edge("reason", "critic")
    
    # Critic Logic (Approve or Reject/Loop)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "router": "router",  # Go back to router to increment attempt (re-plan if needed)
            "report": "report"
        }
    )
    
    graph.add_edge("report", END)
    
    # Compile
    app = graph.compile()
    
    logger.info("RxGuard graph compiled successfully")
    
    return app


# Global instance for import
rxguard_app = build_graph()