"""FastAPI entry point for RxGuard clinical agent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agentic.graph.builder import rxguard_app
from src.agentic.state.schemas import RxGuardState
from src.agentic.utils.logging_config import configure_logging, get_logger

# Configure logging on startup
configure_logging("INFO")
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RxGuard API",
    description="Clinical medication safety checking agent",
    version="0.1.0"
)


# === REQUEST/RESPONSE MODELS ===

class ClinicalNoteRequest(BaseModel):
    """Request body for clinical note submission."""
    
    raw_note: str = Field(
        ...,
        min_length=10,
        description="Clinical note text (e.g., '65M, Stage 3 CKD, Ibuprofen 800mg TID')",
        example="65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days."
    )


class ClinicalReportResponse(BaseModel):
    """Response with clinical safety report."""
    
    alert_level: str = Field(..., description="CRITICAL, WARNING, or INFO")
    patient_context: str = Field(..., description="Summarized patient demographics and conditions")
    identified_risk: str = Field(..., description="Risk summary and mechanism")
    guideline_evidence: list[str] = Field(..., description="Citations from guidelines")
    confidence: str = Field(..., description="High, Moderate, or Low")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str


# === API ENDPOINTS ===

@app.post(
    "/check",
    response_model=ClinicalReportResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Check medication safety",
    description="Submit a clinical note and receive a safety analysis report."
)
async def check_medication(request: ClinicalNoteRequest):
    """Run clinical safety check on a patient note."""
    logger.info("Received clinical note", note_preview=request.raw_note[:50])
    
    # Initialize state
    initial_state: RxGuardState = {
        "raw_note": request.raw_note,
        "patient_profile": None,
        "proposed_medication": None,
        "confidence": None,
        "retrieved_guidelines": None,
        "risk_analysis": None,
        "safety_flag": None,
        "final_report": None
    }
    
    try:
        # Run graph
        result = rxguard_app.invoke(initial_state)
        
        # Check if pipeline stopped early (low confidence)
        if result["final_report"] is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract clinical information with sufficient confidence "
                       f"(confidence: {result['confidence']}). Please provide clearer note."
            )
        
        # Return report
        report = result["final_report"]
        logger.info("Report generated", alert_level=report["alert_level"])
        
        return ClinicalReportResponse(**report)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get(
    "/health",
    summary="Health check",
    description="Check if API is running."
)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "rxguard"}


# === RUN (for development) ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)