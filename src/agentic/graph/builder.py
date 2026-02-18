"""Graph Builder: Compiles LangGraph with all nodes and edges."""

from langgraph.graph import StateGraph, START, END

from src.agentic.graph.edges.conditionals import confidence_gate
from src.agentic.graph.nodes.extract_profile import extract_patient_profile
from src.agentic.graph.nodes.guideline_retrieval import guideline_retrieval_node
from src.agentic.graph.nodes.risk_reasoning import risk_reasoning_node
from src.agentic.graph.nodes.safety_critic import safety_critic_node
from src.agentic.graph.nodes.final_report import final_report_node
from src.agentic.graph.nodes.router import router_node, route_after_router, route_after_critic
from src.agentic.state.schemas import RxGuardState
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)


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
    graph.add_node("retrieve", guideline_retrieval_node)
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
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "report": "report"
        }
    )
    
    # Linear steps in the loop
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "critic")
    
    # Critic Logic (Approve or Reject/Loop)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "router": "router",  # Go back to router to increment attempt
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