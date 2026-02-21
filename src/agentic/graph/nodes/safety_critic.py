"""Safety Critic Node: Validates reasoning and flags safety concerns."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.agentic.agents.base import get_llm
from src.agentic.state.schemas import CritiqueResult, RxGuardState
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)

# Setup critic chain
critic_parser = PydanticOutputParser(pydantic_object=CritiqueResult)

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior clinical pharmacist reviewing a junior's work."),
    ("human",
     """TASK:
Review the patient context, proposed medication, and risk analysis.
Determine if the Risk Analysis is ACCURATE and SUFFICIENT.

CRITIQUE GUIDELINES:
1. **APPROVE** if:
   - The analysis correctly identifies the key risks (e.g., contraindications, major interactions).
   - The risk level is appropriate (e.g., "High" for contraindications).
   - Evidence is cited.
   
2. **REJECT** if:
   - The analysis misses a critical contraindication or warning.
   - The risk level is too low (e.g., "Low" for a contraindication).
   - The explanation is factually incorrect.

OUTPUT INSTRUCTIONS:
- If APPROVE: Set decision to "APPROVE" and populate safety_flag with the appropriate level ("info", "warning", "critical") and reason.
- If REJECT: Set decision to "REJECT" and provide specific feedback on what to fix.

PATIENT CONTEXT:
{patient_context}

PROPOSED MEDICATION:
{medication_context}

RISK ANALYSIS:
{risk_analysis}

JSON FORMAT INSTRUCTIONS:
{format_instructions}
""")
])

# Initialize LLM and chain
llm = get_llm()
critic_chain = critic_prompt | llm | critic_parser


def safety_critic_node(state: RxGuardState, config: RunnableConfig) -> RxGuardState:
    """Review and critique the risk analysis.
    
    Args:
        state: Current graph state
        config: RunnableConfig for callbacks
        
    Returns:
        Updated state with critique_feedback and safety_flag
    """
    logger.info("--- SAFETY CRITIC (Reflexive) ---")
    
    result = critic_chain.invoke({
        "patient_context": state["patient_profile"],
        "medication_context": state["proposed_medication"],
        "risk_analysis": state["risk_analysis"],
        "format_instructions": critic_parser.get_format_instructions()
    }, config=config)
    
    # Update state
    state["critique_feedback"] = result.feedback
    if result.decision == "APPROVE":
        # Only set safety flag if approved (otherwise we re-reason)
        state["safety_flag"] = result.safety_flag.model_dump() if result.safety_flag else None
        # Clear feedback on approval to prevent loops
        state["critique_feedback"] = "APPROVE" 
    else:
        logger.warning(f"CRITIQUE REJECTED: {result.feedback}")
        # Reset risk analysis if rejected to force re-generation? 
        # Actually, we keep it to show history, but the loop will overwrite it.
        pass

    logger.info(f"Critic Decision: {result.decision}")
    
    return state