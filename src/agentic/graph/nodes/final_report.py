"""Final Report Node: Generates structured clinical report."""

from src.agentic.state.schemas import RxGuardState
from src.agentic.utils.logging_config import get_logger, log_clinical_event

logger = get_logger(__name__)


def generate_clinical_report(patient_profile, proposed_medication, risk_analysis, safety_flag):
    """Generate final clinical report dictionary."""
    return {
        "alert_level": safety_flag["level"].upper(),
        "patient_context": (
            f"{patient_profile.get('age')} year old "
            f"{patient_profile.get('sex')} with "
            + ", ".join(patient_profile.get("conditions", []))
        ),
        "identified_risk": (
            f"{risk_analysis['summary']} "
            f"Mechanism: {risk_analysis['mechanism']}."
        ),
        "guideline_evidence": [
            f"{e['source']} (page {e['page']})"
            for e in risk_analysis["evidence"]
        ],
        "confidence": risk_analysis["risk_level"].capitalize()
    }


def final_report_node(state: RxGuardState) -> RxGuardState:
    """Generate final clinical report from all previous analysis.
    
    Args:
        state: Complete graph state with all previous node outputs
        
    Returns:
        Updated state with final_report
    """
    logger.info("--- FINAL REPORT GENERATION ---")
    
    # Generate report
    report = generate_clinical_report(
        state["patient_profile"],
        state["proposed_medication"],
        state["risk_analysis"],
        state["safety_flag"]
    )
    
    state["final_report"] = report
    
    # Log clinical event for audit trail
    log_clinical_event(
        logger=logger,
        event_type="clinical_report_generated",
        patient_context=state["patient_profile"],
        medication_context=state["proposed_medication"],
        risk_analysis=state["risk_analysis"],
        safety_flag=state["safety_flag"],
        alert_level=report["alert_level"]
    )
    
    logger.info(
        "Report generated",
        alert_level=report["alert_level"],
        patient=report["patient_context"][:50] + "..."
    )
    
    return state