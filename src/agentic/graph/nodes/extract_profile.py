"""Node 1: Clinical context extraction node: Parses raw clinical notes into structured data."""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.agentic.state.schemas import RxGuardState, ExtractionResult
from config.settings import settings
from src.agentic.utils import get_logger, log_clinical_event

logger = get_logger(__name__)

# === CONFIGURATION ===
CONFIDENCE_THRESHOLD = 0.75

# Initialize LLM
llm = ChatGroq(model=settings.model_name,
               temperature=settings.temperature,
               api_key=settings.GROQ_API_KEY)

# === PROMPT ENGINEERING ===
parser = PydanticOutputParser(pydantic_object=ExtractionResult)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a clinical information extraction system.

TASK:
Extract structured medical facts from the input note.

RULES:
- Do NOT provide medical advice
- Do NOT infer unstated facts
- If information is missing, use null
- Output must strictly match the JSON schema
- No explanations, no prose

NORMALIZATION RULES:
- Normalize sex to: "male" or "female"
- Normalize CKD stages to: "Chronic Kidney Disease Stage X"
- If CKD is present, include "renal impairment" in risk_factors
- Chronic diseases → conditions
- Pain, discomfort → symptoms (do not include in conditions)

NOTE:
This output will be used by downstream safety systems.
Accuracy is critical for patient safety.
"""),
    ("human",
     "Clinical note:\n{note}\n\n"
     "Return JSON matching this schema:\n{format_instructions}")
])

# === EXTRACTION CHAIN ===
extractor_chain = prompt | llm | parser

# === HELPER FUNCTIONS ===
def run_extraction(note: str) -> ExtractionResult:
    """Execute extraction chain with validation."""
    logger.info("--- CLINICAL EXTRACTION ---")
    
    result = extractor_chain.invoke({
        "note": note,
        "format_instructions": parser.get_format_instructions()
    })
    
    confidence = result.extraction_confidence
    logger.info(f"Confidence Score: {confidence:.1%}")
    
    if result.extraction_confidence < CONFIDENCE_THRESHOLD:
        raise ValueError(
            f"Extraction confidence too low: {result.extraction_confidence:.2f} "
            f"(threshold: {CONFIDENCE_THRESHOLD})"
        )
    
    return result


# === NODE FUNCTION ===
def extract_patient_profile(state: RxGuardState) -> RxGuardState:
    """Node function to extract patient profile from clinical note."""
    logger.info("--- EXTRACTING PATIENT PROFILE ---")
    

    try:
        # Validate input
        raw_note = state.get("raw_note", "")
        if not raw_note or not raw_note.strip():
            raise ValueError("Empty clinical note provided")
        
        logger.info(f"Processing note ({len(raw_note)} characters)")
        
        # Run extraction
        extraction = run_extraction(raw_note)
        
        # Log extracted data summary
        logger.info(
            f"✅ Extracted: "
            f"{len(extraction.patient_profile.conditions)} conditions, "
            f"{len(extraction.patient_profile.risk_factors)} risk factors"
        )
        
        if extraction.proposed_medication.drug_name:
            logger.info(
                f"   Medication: {extraction.proposed_medication.drug_name} "
                f"{extraction.proposed_medication.dose_mg_per_unit}mg"
            )
        
        # Update state with structured data
        return {
            **state,
            "patient_profile": extraction.patient_profile.model_dump(),
            "proposed_medication": extraction.proposed_medication.model_dump(),
            "confidence": extraction.extraction_confidence,
        }
    except Exception as e:
        logger.error(f"Error in patient profile extraction: {e}")
        raise e

