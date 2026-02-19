"""RxGuard - Pure Streamlit Application."""

import streamlit as st
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="RxGuard - Clinical Safety Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
import sys
import asyncio
import os
# Add project root to sys.path to allow absolute imports from 'src'
sys.path.append(str(Path(__file__).parent.parent))

from src.agentic.state.schemas import RxGuardState, create_initial_state
from src.agentic.utils.logging_config import configure_logging, get_logger

# Setup logging
configure_logging("INFO")
logger = get_logger(__name__)

# --- FIXES FOR STREAMLIT + LANGCHAIN/TORCH ISSUES ---

# 1. Windows Asyncio Fix
# Streamlit on Windows requires the SelectorEventLoopPolicy for async features
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 2. Torch Class Watcher Fix
# Monkeypatch torch.classes to have a __path__ attribute so Streamlit's
# file watcher doesn't crash when it tries to iterate over it.
try:
    import torch
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except ImportError:
    pass

# ----------------------------------------------------

@st.cache_resource
def get_rxguard_app():
    """Lazy load the graph to prevent file watcher issues."""
    from src.agentic.graph.builder import rxguard_app
    return rxguard_app


def run_clinical_analysis(raw_note: str):
    """Run full graph analysis."""
    try:
        # Create initial state
        state = create_initial_state(raw_note)
        
        # Get cached app
        app = get_rxguard_app()
        
        final_state = state
        
        # Use a status container for live updates
        # Create a placeholder for the thought process *outside* the status container first?
        # No, StreamlitCallbackHandler works best with a container.
        
        st_callback = None
            
        # Run graph with streaming
        # app.stream yields dictionaries keyed by node name: {'node': state_update}
        
        # Create a container for the live thought process
        thought_container = st.container()
        
        with thought_container:
            # Initialize the StreamlitCallbackHandler
            # We want it to render inside an expander or container
            from langchain_community.callbacks import StreamlitCallbackHandler
            st_callback = StreamlitCallbackHandler(
                st.container(), 
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
                thought_labeler=None # Default labeler
            )

        # Run with callback
        config = {"callbacks": [st_callback], "recursion_limit": 50}
        
        # We still want the status indicator for high-level progress
        with st.status("🚀 RxGuard Agents Active...", expanded=False) as status:
            
            for output in app.stream(state, config=config):
                for node_name, state_update in output.items():
                    # Update our local view of state
                    final_state.update(state_update)
                    
                    if node_name == "extract_patient_profile":
                        status.write("✅ Patient Profile Extracted")
                    
                    elif node_name == "planner":
                        cnt = len(state_update.get("plan", []))
                        status.write(f"✅ Research Plan Created ({cnt} steps)")
                        
                    elif node_name == "research_agent":
                        # Check which step just finished
                        idx = final_state.get("current_step_index", 0)
                        # The update happens *after* the step is done, so idx might be next step
                        # But we can look at the log
                        log = state_update.get("research_log", [])
                        if log:
                            last_log = log[-1]
                            # Clean up log string for display
                            display_log = last_log.split(":")[-1].strip()
                            status.write(f"🔎 Research: {display_log}")
                            
                    elif node_name == "risk_reasoning":
                        risk = state_update.get("risk_analysis", {}).get("risk_level", "Unknown")
                        status.write(f"🤔 Risk Reasoning Complete (Level: {risk})")
                        
                    elif node_name == "safety_critic":
                        decision = state_update.get("critique_feedback", "")
                        if decision == "APPROVE":
                            status.write("🛡️ Safety Critic: APPROVED")
                        else:
                            status.write(f"🛡️ Safety Critic: REJECTED (Looping back...)")
                            
            status.update(label="✅ Clinical Analysis Complete", state="complete", expanded=False)
        
        return final_state
    
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        st.error(f"❌ Analysis error: {str(e)}")
        return None


def render_thought_process(result: dict):
    """Render the agent's internal thought process (Research & Critique)."""
    
    with st.expander("🧠 Agent Thought Process (Trace)", expanded=True):
        # 1. Attempts
        attempts = result.get("attempts", 0)
        st.caption(f"Reasoning Cycles: {attempts}")
        
        # 2. Plan Execution (Structured)
        plan = result.get("plan", [])
        if plan:
            st.markdown("### 📋 Research Plan & Execution")
            for step in plan:
                # Handle Pydantic model or dict
                if hasattr(step, "model_dump"):
                    s_dict = step.model_dump()
                else:
                    s_dict = step
                
                status_icon = "✅" if s_dict.get("status") == "complete" else "⏳"
                with st.container():
                    st.markdown(f"**{status_icon} Step {s_dict.get('id')}:** {s_dict.get('description')}")
                    if s_dict.get("result"):
                        st.caption(f"📝 *Findings:* {s_dict.get('result')}")
                    st.divider()
        else:
             # Fallback to legacy log if plan is missing
            logs = result.get("research_log", [])
            if logs:
                st.markdown("### 🔎 Research History (Legacy)")
                for i, log in enumerate(logs, 1):
                    st.markdown(f"**Step {i}:** `{log}`")
        
        # 3. Critique Feedback
        feedback = result.get("critique_feedback")
        if feedback:
            if feedback == "APPROVE":
                 st.success("🛡️ Safety Critic: Analysis Approved ✅")
            else:
                 st.error(f"🛡️ Safety Critic: Rejected previous analysis. Feedback: {feedback}")


def render_report(result: dict):
    """Render clinical report."""
    report = result.get("final_report")
    
    if not report:
        st.warning("⚠️ Could not generate report. Low extraction confidence?")
        return
    
    # Alert level with color coding
    alert_colors = {
        "CRITICAL": ("🔴", "#dc3545", "#f8d7da"),
        "WARNING": ("🟡", "#ffc107", "#fff3cd"),
        "INFO": ("🟢", "#17a2b8", "#d1ecf1")
    }
    
    emoji, badge_color, bg_color = alert_colors.get(
        report["alert_level"], 
        ("⚪", "#6c757d", "#f8f9fa")
    )
    
    # Alert header
    st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid {badge_color};
            margin-bottom: 20px;
        ">
            <h2 style="margin:0;color:{badge_color};">
                {emoji} {report["alert_level"]} ALERT
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Render Thought Process First
    render_thought_process(result)
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Patient Context")
        st.info(report["patient_context"])
        
        st.subheader("✅ Confidence")
        conf_emoji = "🟢" if report.get("confidence") == "High" else "🟡" if report.get("confidence") == "Moderate" else "🔴"
        st.success(f"{conf_emoji} **{report.get('confidence', 'Unknown')}**")
    
    with col2:
        st.subheader("⚠️ Identified Risk")
        st.warning(report["identified_risk"])
    
    # Evidence section
    st.subheader("📚 Guideline Evidence")
    if report.get("guideline_evidence"):
        for i, evidence in enumerate(report["guideline_evidence"], 1):
            with st.container():
                st.markdown(f"**{i}. {evidence}**")
                st.divider()
    else:
        st.info("No specific citations available for this finding.")
    
    # Raw data expander
    with st.expander("🔧 View Technical Details"):
        st.json({
            "patient_profile": result.get("patient_profile"),
            "proposed_medication": result.get("proposed_medication"),
            "risk_analysis": result.get("risk_analysis"),
            "safety_flag": result.get("safety_flag"),
            "final_report": report
        })


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
        <h1 style="color:#1f77b4;margin-bottom:0;">🛡️ RxGuard</h1>
        <p style="color:#666;font-size:1.1em;">
            AI-Powered Clinical Medication Safety Checker
        </p>
        <hr>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.caption("Version 3.0.0")
        st.info("""
            **RxGuard** analyzes clinical notes for medication safety issues using:
            - LangGraph multi-agent orchestration
            - RAG with clinical guidelines
            - Groq LLM (Llama 3.3 70B)
        """)
        
        st.header("📖 Examples")
        examples = [
            "65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days.",
            "72F, Atrial fibrillation on Warfarin, headache. Plan: Aspirin 325mg.",
            "45F, Type 2 Diabetes, HbA1c 8.2%. Plan: Metformin 500mg BID.",
            "28F, 16 weeks pregnant, acne. Plan: Isotretinoin 20mg daily.",
        ]
        
        for i, ex in enumerate(examples, 1):
            if st.button(f"Example {i}", key=f"ex_{i}"):
                st.session_state.note_input = ex
                st.rerun()
        
        st.header("⚙️ Settings")
        st.caption("Model: llama-3.3-70b-versatile")
        st.caption("Temp: 0.1 (clinical precision)")
        
        # Check vectorstore
        vs_path = Path("data/vectorstore/guidelines_v1")
        if vs_path.exists():
            st.success("✅ Vectorstore ready")
        else:
            st.warning("⚠️ Will create vectorstore on first run")
    
    # Main input area
    st.subheader("📝 Enter Clinical Note")
    
    # Get example from session state if set
    default_note = st.session_state.get("note_input", 
        "65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days.")
    
    raw_note = st.text_area(
        "Clinical note (patient info + prescribed medication):",
        value=default_note,
        height=120,
        placeholder="e.g., 65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days."
    )
    
    # Buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.note_input = ""
            st.rerun()
    
    # Results section
    st.markdown("---")
    
    if analyze and raw_note:
        result = run_clinical_analysis(raw_note)
        if result:
            render_report(result)
    elif analyze and not raw_note:
        st.warning("⚠️ Please enter a clinical note first.")


if __name__ == "__main__":
    main()