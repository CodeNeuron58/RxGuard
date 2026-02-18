"""Final Report Node: Generates structured clinical report."""

from src.agentic.state.schemas import RxGuardState
from src.agentic.utils.logging_config import get_logger, log_clinical_event

logger = get_logger(__name__)


def generate_clinical_report(patient_profile, proposed_medication, risk_analysis, safety_flag):
    """Generate final clinical report dictionary."""
    
    # Handle None cases (e.g. forced exit from loop)
    if not safety_flag:
        flag_level = "WARNING"
        flag_reason = "Analysis inconclusive: Maximum research attempts reached without final approval."
    else:
        flag_level = safety_flag.get("level", "INFO").upper()
        # flag_reason = safety_flag.get("reason", "") # Not used in current template but good to have
        
    if not risk_analysis:
        risk_summary = "Risk analysis could not be completed."
        risk_mechanism = "N/A"
        evidence_list = []
        risk_level = "Unknown"
    else:
        risk_summary = risk_analysis.get("summary", "")
        risk_mechanism = risk_analysis.get("mechanism", "N/A")
        risk_level = risk_analysis.get("risk_level", "Unknown")
        
        # Handle evidence list safely
        evidence_raw = risk_analysis.get("evidence", [])
        evidence_list = []
        for e in evidence_raw:
            if isinstance(e, dict):
                evidence_list.append(f"{e.get('source', 'Unknown')} (page {e.get('page', '?')})")
            elif hasattr(e, 'source'):
                evidence_list.append(f"{e.source} (page {e.page})")
            else:
                evidence_list.append(str(e))

        # ESCALATION: If risk is HIGH, always flag as CRITICAL
        if risk_level == "High":
            flag_level = "CRITICAL"

    return {
        "alert_level": flag_level,
        "patient_context": (
            f"{patient_profile.get('age', '?')} year old "
            f"{patient_profile.get('sex', '?')} with "
            + ", ".join(patient_profile.get("conditions", []))
        ) if patient_profile else "Patient profile unknown",
        
        "identified_risk": (
            f"{risk_summary} "
            f"Mechanism: {risk_mechanism}."
        ),
        "guideline_evidence": evidence_list,
        "confidence": risk_level.capitalize()
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
        state.get("patient_profile"),
        state.get("proposed_medication"),
        state.get("risk_analysis"),
        state.get("safety_flag")
    )
    
    state["final_report"] = report
    
    # Log clinical event for audit trail
    log_clinical_event(
        logger=logger,
        event_type="clinical_report_generated",
        patient_context=state.get("patient_profile"),
        medication_context=state.get("proposed_medication"),
        risk_analysis=state.get("risk_analysis"),
        safety_flag=state.get("safety_flag"),
        alert_level=report["alert_level"]
    )
    
    logger.info(
        "Report generated",
        alert_level=report["alert_level"],
        patient=report["patient_context"][:50] + "..."
    )
    
    return state