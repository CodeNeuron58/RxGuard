"""Streamlit UI for RxGuard Clinical Agent."""

import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="RxGuard - Clinical Safety Checker",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .alert-critical {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .alert-warning {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    .alert-info {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
    }
    .evidence-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_medication_api(raw_note: str):
    """Call FastAPI backend."""
    try:
        response = requests.post(
            "http://localhost:8000/check",
            json={"raw_note": raw_note},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to API. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.json().get('detail', str(e))}")
        return None


def render_alert_badge(level: str):
    """Render colored alert badge."""
    colors = {
        "CRITICAL": ("🔴", "#dc3545"),
        "WARNING": ("🟡", "#ffc107"),
        "INFO": ("🟢", "#17a2b8")
    }
    emoji, color = colors.get(level, ("⚪", "#6c757d"))
    return f"<span style='background-color:{color};color:white;padding:5px 15px;border-radius:20px;font-weight:bold;'>{emoji} {level}</span>"


def main():
    # Header
    st.markdown("<p class='main-header'>🛡️ RxGuard</p>", unsafe_allow_html=True)
    st.markdown("*AI-Powered Clinical Medication Safety Checker*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **RxGuard** analyzes clinical notes for medication safety issues.
        
        **How to use:**
        1. Enter a clinical note
        2. Click 'Analyze'
        3. Review the safety report
        
        **Example notes:**
        - '65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days.'
        - '45F, Diabetes, Metformin 500mg BID'
        """)
        
        st.header("⚙️ Settings")
        api_url = st.text_input("API URL", "http://localhost:8000")
        st.caption("Make sure FastAPI is running!")
    
    # Main input
    st.subheader("📝 Clinical Note")
    
    example_note = "65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days."
    
    raw_note = st.text_area(
        "Enter patient information and prescribed medication:",
        value=example_note,
        height=150,
        placeholder="e.g., 65M, Stage 3 CKD, severe back pain. Plan: Ibuprofen 800mg TID x5 days."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=False):
            st.rerun()
    
    st.markdown("---")
    
    # Results
    if analyze_btn and raw_note:
        with st.spinner("🔬 Analyzing clinical note... (this may take 10-20 seconds)"):
            result = check_medication_api(raw_note)
        
        if result:
            st.subheader("📊 Safety Report")
            
            # Alert level with styling
            alert_class = result['alert_level'].lower()
            st.markdown(f"""
            <div class='alert-{alert_class}'>
                <h3>{render_alert_badge(result['alert_level'])}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Patient context
            st.markdown("#### 👤 Patient Context")
            st.info(result['patient_context'])
            
            # Identified risk
            st.markdown("#### ⚠️ Identified Risk")
            st.warning(result['identified_risk'])
            
            # Guideline evidence
            st.markdown("#### 📚 Guideline Evidence")
            if result['guideline_evidence']:
                for evidence in result['guideline_evidence']:
                    st.markdown(f"""
                    <div class='evidence-box'>
                        📄 {evidence}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No specific guideline citations available")
            
            # Confidence
            st.markdown("#### ✅ Analysis Confidence")
            confidence_emoji = "🟢" if result['confidence'] == "High" else "🟡" if result['confidence'] == "Moderate" else "🔴"
            st.success(f"{confidence_emoji} **{result['confidence']}**")
            
            # Raw JSON (expandable)
            with st.expander("🔧 View Raw JSON"):
                st.json(result)
    
    elif analyze_btn and not raw_note:
        st.warning("⚠️ Please enter a clinical note first.")


if __name__ == "__main__":
    main()