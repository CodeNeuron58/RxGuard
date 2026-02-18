
"""Research Agent Node: Executes the current plan step and extracts structured evidence."""

from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import settings
from src.agentic.agents.base import get_llm
from src.agentic.state.schemas import RxGuardState, EvidenceItem
from src.agentic.utils.logging_config import get_logger
from src.agentic.graph.nodes.guideline_retrieval import get_vectorstore # Reuse existing FAISS logic

logger = get_logger(__name__)

# === 1. Query Generation for Current Step ===
query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Generate a precise search query for the given task."),
    ("human", 
     "Task: {step_description}\n"
     "Patient Context: {patient_context}\n"
     "Drug: {drug}\n\n"
     "Output ONLY the search query string.")
])

query_gen_chain = query_gen_prompt | get_llm() | StrOutputParser()

# === 2. Evidence Extraction (The 'Agentic' part) ===
class ExtractedEvidence(BaseModel):
    """List of evidence items extracted from text."""
    items: List[EvidenceItem] = Field(description="List of relevant facts found.")

extractor_parser = PydanticOutputParser(pydantic_object=ExtractedEvidence)

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a meticulous medical researcher. Your goal is to extract evidence relevant to the current research task.\n"
     "Review the retrieved text snippets and extract facts that answer the research task.\n"
     "If a snippet is irrelevant, ignore it."),
    ("human",
     "Current Research Task: {step_description}\n\n"
     "Retrieved Snippets:\n{snippets}\n\n"
     "Extract any evidence found as a list of structured items.\n"
     "{format_instructions}")
])

extractor_chain = extractor_prompt | get_llm() | extractor_parser

def research_agent_node(state: RxGuardState) -> RxGuardState:
    """Execute the current step in the research plan."""
    logger.info("--- RESEARCH AGENT ---")
    
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)
    
    if current_index >= len(plan):
        logger.info("All plan steps complete.")
        return state
        
    current_step = plan[current_index]
    logger.info(f"Executing Step {current_step.id}: {current_step.description}")
    
    # Update step status
    current_step.status = "in_progress"
    
    # RATE LIMITING: Sleep to avoid 429s from Groq
    import time
    time.sleep(2.0)
    
    # 1. Generate Query
    patient_profile = state["patient_profile"]
    proposed_medication = state["proposed_medication"]
    
    query = query_gen_chain.invoke({
        "step_description": current_step.description,
        "patient_context": str(patient_profile),
        "drug": proposed_medication.get("drug_name")
    })
    
    # 2. Search (Reuse existing FAISS)
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=settings.TOP_K_RETRIEVAL)
    
    snippets = "\n\n".join([f"Source: {r.metadata.get('source')}\nContent: {r.page_content}" for r in results])
    
    # 3. Extract Evidence
    try:
        extraction = extractor_chain.invoke({
            "step_description": current_step.description,
            "snippets": snippets,
            "format_instructions": extractor_parser.get_format_instructions()
        })
        
        # Add to global evidence log
        current_evidence = state.get("evidence", [])
        current_evidence.extend(extraction.items)
        state["evidence"] = current_evidence
        
        # Mark step complete
        current_step.status = "complete"
        current_step.result = f"Found {len(extraction.items)} evidence items."
        
        # Update Log for UI
        log_entry = f"Step {current_step.id}: {current_step.description} -> Found {len(extraction.items)} facts."
        current_log = state.get("research_log", [])
        current_log.append(log_entry)
        state["research_log"] = current_log
        
        # Increment index for next step
        state["current_step_index"] = current_index + 1
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        current_step.status = "complete" # Move on anyway
        state["current_step_index"] = current_index + 1
        
    return state
