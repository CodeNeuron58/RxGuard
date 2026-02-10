"""Risk Reasoning Node: Analyzes clinical risks using retrieved guidelines."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agentic.agents.base import get_llm
from src.agentic.state.schemas import RiskAnalysis, RxGuardState
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)

# Setup risk reasoning chain
risk_parser = PydanticOutputParser(pydantic_object=RiskAnalysis)

risk_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert clinical pharmacist and risk reasoning system.\n"
     "Your goal is to evaluate the safety of a proposed medication for a specific patient, "
     "using ONLY the provided clinical guidelines."),
    ("human",
     "Analyze the clinical risk of the proposed medication for this patient.\n\n"
     "Strictly follow these steps:\n"
     "1. Review the Patient Context (conditions, risk factors, age).\n"
     "2. Review the Proposed Medication (dose, frequency).\n"
     "3. Search the Guideline Excerpts for ANY contraindications, warnings, or dose adjustments "
     "relevant to this patient's specific conditions or demographics.\n"
     "4. If a risk is found, explain the physiological mechanism (e.g., 'NSAIDs constrict afferent arterioles...').\n"
     "5. Assign a risk level (low, moderate, high).\n"
     "6. Cite the specific guideline source and page number for every claim.\n\n"
     "If the medication is safe based on the provided guidelines, state 'Low' risk.\n"
     "If the guidelines do not mention the medication or condition, state 'Low' risk but note the lack of specific evidence.\n\n"
     "Patient Context:\n{patient_context}\n\n"
     "Proposed Medication:\n{medication_context}\n\n"
     "Guideline Excerpts:\n{guideline_text}\n\n"
     "Output Format:\n{format_instructions}")
])

# Initialize LLM and chain
llm = get_llm()
risk_chain = risk_prompt | llm | risk_parser


def risk_reasoning_node(state: RxGuardState) -> RxGuardState:
    """Analyze clinical risks based on patient, medication, and guidelines.
    
    Args:
        state: Current graph state with retrieved_guidelines, patient_profile, 
               and proposed_medication
        
    Returns:
        Updated state with risk_analysis
    """
    logger.info("--- RISK REASONING ---")
    
    # Format guideline excerpts for the prompt
    guideline_text = "\n\n".join(
        f"Source: {g['source']}, Page: {g['page']}\n{g['content']}"
        for g in state["retrieved_guidelines"]
    )
    
    # Run risk analysis
    risk = risk_chain.invoke({
        "patient_context": state["patient_profile"],
        "medication_context": state["proposed_medication"],
        "guideline_text": guideline_text,
        "format_instructions": risk_parser.get_format_instructions()
    })
    
    # Store result as dict
    state["risk_analysis"] = risk.model_dump()
    
    logger.info(
        "Risk analysis complete",
        risk_level=risk.risk_level,
        summary=risk.summary[:100] + "..."
    )
    
    return state