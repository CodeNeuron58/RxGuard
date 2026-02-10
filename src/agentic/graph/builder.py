"""Graph Builder: Compiles LangGraph with all nodes and edges."""

from langgraph.graph import StateGraph, START, END

from src.agentic.graph.edges.conditionals import confidence_gate
from src.agentic.graph.nodes.context_extraction import context_extraction_node
from src.agentic.graph.nodes.guideline_retrieval import guideline_retrieval_node
from src.agentic.graph.nodes.risk_reasoning import risk_reasoning_node
from src.agentic.graph.nodes.safety_critic import safety_critic_node
from src.agentic.graph.nodes.final_report import final_report_node
from src.agentic.state.schemas import RxGuardState
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """Build and compile the RxGuard clinical agent graph.
    
    Returns:
        Compiled LangGraph application
    """
    logger.info("Building RxGuard graph...")
    
    # Initialize graph with state schema
    graph = StateGraph(RxGuardState)
    
    # Add nodes
    graph.add_node("extract", context_extraction_node)
    graph.add_node("retrieve", guideline_retrieval_node)
    graph.add_node("reason", risk_reasoning_node)
    graph.add_node("critic", safety_critic_node)
    graph.add_node("report", final_report_node)
    
    # Add edges
    graph.add_edge(START, "extract")
    
    # Conditional edge: confidence check
    graph.add_conditional_edges(
        "extract",
        confidence_gate,
        {
            "retrieve": "retrieve",
            "stop": END
        }
    )
    
    # Linear flow after confidence check passes
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "critic")
    graph.add_edge("critic", "report")
    graph.add_edge("report", END)
    
    # Compile
    app = graph.compile()
    
    logger.info("RxGuard graph compiled successfully")
    
    return app


# Global instance for import
rxguard_app = build_graph()