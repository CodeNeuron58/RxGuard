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
    """The canonical state passed through every node in your clinical agent."""
    # === INPUT (set at entry) ===
    raw_note: str = Field(
        default="", 
        description="Original clinical note text from user."
    )
    # === EXTRACTED DATA (populated by understanding node) ===
    patient_profile: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Structured patient data extracted from raw_note."
    )
    proposed_medication: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Medication recommendations with safety checks."
    )
    confidence: Optional[float] = Field(
        default=None, 
        description="Confidence level of the extraction."
    )
    # === WORK PRODUCTS (populated by medication node) ===
    retrieved_guidelines: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Guidelines retrieved from RAG."
    )
    # === SAFETY & VALIDATION (populated by risk node) ===
    risk_analysis: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Risk analysis of proposed medication."
    )
    # === SAFETY & VALIDATION (populated by risk node) ===
    safety_flag: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Safety flag from safety critic node."
    )
    # === OUTPUT (populated by finalizer node) ===
    final_report: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Final formatted response for user."
    )

state = RxGuardState()