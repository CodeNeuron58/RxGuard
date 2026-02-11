"""State schemas: Graph state and LLM output contracts."""

from typing import Optional, List, Dict, Any, Literal
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
    summary: str
    mechanism: str
    evidence: List[EvidenceCitation]
    risk_level: Literal["low", "moderate", "high"]


class SafetyFlag(BaseModel):
    """Safety flag from safety critic node."""
    level: Literal["info", "warning", "critical"]
    reason: str


# === GRAPH STATE SCHEMA (what flows through nodes) ===

class RxGuardState(TypedDict):
    """State passed through clinical agent nodes."""
    # Input
    raw_note: str
    
    # Extracted data
    patient_profile: dict[str, Any] | None
    proposed_medication: dict[str, Any] | None
    confidence: float | None
    
    # Work products
    retrieved_guidelines: list[dict[str, Any]] | None
    
    # Safety validation
    risk_analysis: dict[str, Any] | None
    safety_flag: dict[str, Any] | None
    
    # Output
    final_report: dict[str, Any] | None


def create_initial_state(raw_note: str) -> RxGuardState:
    """Create initial state with default None values."""
    return {
        "raw_note": raw_note,
        "patient_profile": None,
        "proposed_medication": None,
        "confidence": None,
        "retrieved_guidelines": None,
        "risk_analysis": None,
        "safety_flag": None,
        "final_report": None
    }
