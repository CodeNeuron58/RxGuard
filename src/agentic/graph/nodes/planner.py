
"""Planner Node: Decomposes the safety check into a list of research steps."""

from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agentic.agents.base import get_llm
from src.agentic.state.schemas import RxGuardState, PlanStep
from src.agentic.utils.logging_config import get_logger

logger = get_logger(__name__)

class ResearchPlan(BaseModel):
    """The full research plan."""
    steps: List[PlanStep] = Field(description="List of steps to research.")

# 1. Setup Planner Chain
parser = PydanticOutputParser(pydantic_object=ResearchPlan)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a Senior Clinical Pharmacist planning a safety review.\n"
     "Your goal is to break down a medication safety check into distinct, logical research steps.\n"
     "Think about: Contraindications, Dose Adjustments (Renal/Hepatic), Drug-Drug Interactions, and Geriatric Risks."),
    ("human",
     "Create a research plan to verify the safety of {drug} for this patient.\n\n"
     "Patient Profile: {patient_context}\n"
     "Proposed Medication: {medication_context}\n\n"
     "Return a list of specific questions/steps to investigate. Example steps:\n"
     "- 'Check for renal dose adjustments for Gabapentin in Stage 3 CKD.'\n"
     "- 'Investigate interaction between Warfarin and Amiodarone.'\n"
     "- 'Verify safety of anticholinergics in elderly patient with dementia.'\n\n"
     "{format_instructions}")
])

llm = get_llm()
planner_chain = planner_prompt | llm | parser

def planner_node(state: RxGuardState) -> RxGuardState:
    """Generate a research plan based on patient and medication."""
    logger.info("--- PLANNER NODE ---")
    
    # CHECK: If a plan exists and we are NOT starting a new attempt loop (e.g. from Router), skip re-planning.
    # However, the Router increments 'attempts'. If attempts > previous attempts, we need to re-plan?
    # Actually, for this design:
    # 1. First run: Plan is empty -> Generate.
    # 2. Research Loop: Plan exists -> Do NOT Generate.
    # 3. Critic Reject -> Router -> Planner: Plan exists but failed. We need to RE-PLAN or APPEND to plan.
    
    # Simple Logic: If plan is empty, generate it.
    # If plan is NOT empty, but we are here, it means we looped back.
    # If we are in the middle of execution (index < len(plan)), we should NOT be in the Planner Node!
    # The Graph wires Router -> Planner.
    
    existing_plan = state.get("plan", [])
    if existing_plan and len(existing_plan) > 0:
        # If we are here and have a plan, check if we finished it?
        # If we came from Router (Reject), we might want to re-plan.
        # For now, let's assume if we are here, we Re-Plan *only if we are starting a fresh cycle* logic is complex.
        
        # FIX: The "Research Loop" should NOT go back to Planner. It should go Research -> Research.
        # The Graph in builder.py has Research -> Planner conditional. This is WRONG if Planner resets index.
        # We will fix builder.py to loop Research -> Research -> Reason.
        # But if Router sends us here (Retry), we DO want to re-plan.
        
        # So: If we are just looping through steps, we shouldn't be here.
        logger.info("Plan already exists. Checking if we need to re-plan...")
        pass 

    patient_profile = state.get("patient_profile") or {}
    proposed_medication = state.get("proposed_medication") or {}
    
    try:
        # Re-planning logic:
        # Only run LLM if plan is empty OR if we are forcing a re-plan (e.g. attempt > 1)
        if not state.get("plan"): 
            plan_result = planner_chain.invoke({
                "drug": proposed_medication.get("drug_name", "Unknown Drug"),
                "patient_context": str(patient_profile),
                "medication_context": str(proposed_medication),
                "format_instructions": parser.get_format_instructions()
            })
            
            state["plan"] = plan_result.steps
            state["current_step_index"] = 0
            state["evidence"] = [] # Clear old evidence on re-plan
            logger.info(f"Generated Plan with {len(plan_result.steps)} steps.")
        else:
            logger.info("Using existing plan.")

        
        logger.info(f"Generated Plan with {len(plan_result.steps)} steps.")
        for step in plan_result.steps:
            logger.info(f"Step {step.id}: {step.description}")
            
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        # Fallback plan if LLM fails
        state["plan"] = [
            PlanStep(id=1, description=f"Check general contraindications for {proposed_medication.get('drug_name')}", status="pending")
        ]
        
    return state
