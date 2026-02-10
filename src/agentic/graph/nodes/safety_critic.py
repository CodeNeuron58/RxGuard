"""Safety Critic Node: Flags escalation-level safety concerns."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agentic.agents.base import get_llm
from src.agentic.state.schemas import SafetyFlag, RxGuardState
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)

# Setup safety critic chain
safety_parser = PydanticOutputParser(pydantic_object=SafetyFlag)

safety_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clinical safety critic system."),
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

JSON SCHEMA:
{format_instructions}
""")
])

# Initialize LLM and chain
llm = get_llm()
safety_chain = safety_prompt | llm | safety_parser


def safety_critic_node(state: RxGuardState) -> RxGuardState:
    """Review and flag critical safety concerns.
    
    Args:
        state: Current graph state with risk_analysis, patient_profile,
               and proposed_medication
        
    Returns:
        Updated state with safety_flag
    """
    logger.info("--- SAFETY CRITIC ---")
    
    # Run safety check
    flag = safety_chain.invoke({
        "patient_context": state["patient_profile"],
        "medication_context": state["proposed_medication"],
        "risk_analysis": state["risk_analysis"],
        "format_instructions": safety_parser.get_format_instructions()
    })
    
    # Store result as dict
    state["safety_flag"] = flag.model_dump()
    
    # Log critical/warning levels prominently
    if flag.level in ["warning", "critical"]:
        logger.warning(
            f"SAFETY FLAG: {flag.level.upper()}",
            reason=flag.reason,
            patient_context=state["patient_profile"]
        )
    else:
        logger.info(f"Safety check passed: {flag.level}")
    
    return state