"""State schemas: Graph state and LLM output contracts."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from typing import TypedDict


# === LLM OUTPUT SCHEMAS (what LLM returns) ===
class PatientProfile(BaseModel):
    """Structured patient information extracted from clinical notes."""
    age: Optional[int]
    sex: Optional[str]
    conditions: List[str] = []
    risk_factors: List[str] = []


class ProposedMedication(BaseModel):
    """A single proposed medication with details."""
    drug_name: Optional[str]
    dose_mg_per_unit: Optional[int]
    frequency_per_day: Optional[int]
    duration_days: Optional[int]
    total_daily_dose_mg: Optional[int]


class ExtractionResult(BaseModel):
    """Complete extraction result from understanding node."""
    patient_profile: PatientProfile
    proposed_medication: ProposedMedication
    extraction_confidence: float = Field(description="0–1 confidence")


class EvidenceCitation(BaseModel):
    """Citation for guideline evidence."""
    source: str
    page: int


class RiskAnalysis(BaseModel):
    """Risk analysis of proposed medication."""
    risk_level: Literal["Low", "Moderate", "High"] = Field(
        description="The overall risk level (Low/Moderate/High)."
    )
    summary: str
    mechanism: str
    evidence: List[EvidenceCitation]


class SafetyFlag(BaseModel):
    """Safety flag from safety critic node."""
    level: Literal["info", "warning", "critical"]
    reason: str


class CritiqueResult(BaseModel):
    """Structured output from the Safety Critic."""
    decision: Literal["APPROVE", "REJECT"]
    feedback: str = Field(description="Specific feedback on what is missing or incorrect.")
    safety_flag: Optional[SafetyFlag] = None


# === GRAPH STATE SCHEMA (what flows through nodes) ===

class EvidenceItem(BaseModel):
    """Structured evidence found during research."""
    fact: str = Field(description="The specific medical fact found.")
    source: str = Field(description="The source of the fact (e.g. 'KDIGO 2024, p.12').")
    relevance: str = Field(description="Why this fact is relevant to the plan.")
    confidence: str = Field(description="Confidence in this fact (High/Medium/Low).")

class PlanStep(BaseModel):
    """A step in the research plan."""
    id: int
    description: str = Field(description="What needs to be researched.")
    status: Literal["pending", "in_progress", "complete"] = "pending"
    result: Optional[str] = None

class RxGuardState(TypedDict):
    """Reflexive Graph State (Agent Memory)."""
    
    # Input
    raw_note: str
    
    # Context
    patient_profile: dict
    proposed_medication: dict
    confidence: float
    
    # Agency
    plan: List[PlanStep]
    current_step_index: int
    
    # Memory
    evidence: List[EvidenceItem]
    research_log: List[str]  # Human-readable log for UI
    
    # Analysis
    critique_feedback: Optional[str]
    risk_analysis: Optional[dict]
    safety_flag: Optional[dict]
    final_report: Optional[dict]
    
    # Control
    attempts: int
    missing_info_flag: bool


def create_initial_state(raw_note: str) -> RxGuardState:
    """Create initial state with default values."""
    return {
        "raw_note": raw_note,
        "patient_profile": {},
        "proposed_medication": {},
        "confidence": 0.0,
        "plan": [],
        "current_step_index": 0,
        "evidence": [],
        "research_log": [],
        "critique_feedback": None,
        "risk_analysis": None,
        "safety_flag": None,
        "final_report": None,
        "attempts": 0,
        "missing_info_flag": False
    }
