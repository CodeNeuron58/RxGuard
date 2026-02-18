"""Safety Critic Node: Validates reasoning and flags safety concerns."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
Identify whether this case requires escalation due to potential
serious or irreversible harm.

RULES:
- Do NOT repeat the full risk explanation.
- Do NOT give medical advice.
- Do NOT suggest alternatives.
- Flag only significant safety concerns.
- If no escalation is required, return an empty JSON object.
- Output must strictly match the JSON schema.
- Do NOT include any text outside the JSON.

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


def safety_critic_node(state: RxGuardState) -> RxGuardState:
    """Review and critique the risk analysis.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with critique_feedback and safety_flag
    """
    logger.info("--- SAFETY CRITIC (Reflexive) ---")
    
    result = critic_chain.invoke({
        "patient_context": state["patient_profile"],
        "medication_context": state["proposed_medication"],
        "risk_analysis": state["risk_analysis"],
        "format_instructions": critic_parser.get_format_instructions()
    })
    
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