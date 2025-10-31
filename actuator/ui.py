import streamlit as st
from brainflow.board_shim import BoardShim, BoardIds

#returns placeholders 
def render_dashboard():
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Arousal Level")
        arousal_progress = st.progress(0.0)
        arousal_metric = st.metric(label="Arousal Index (rog ratio)", value="0.00")
    with col2:
        st.subheader("System Status")
        status_container = st.empty()
        viability_metric = st.metric(label="Viability Band", value="[0.00, 0.00]")


    return {
        "arousal_progress": arousal_progress,
        "arousal_metric": arousal_metric,
        "status_container": status_container,
        "viability_metric": viability_metric,
    }

def update_main_dashboard(placeholders, arousal, viability_band, in_range, artifact_detected):
    arousal_value = arousal if arousal is not None else 0.0
    AROUSAL_MIN, AROUSAL_MAX = -2.0, 2.0
    normalized_progress = max(0.0, min(1.0, (arousal_value - AROUSAL_MIN) / (AROUSAL_MAX - AROUSAL_MIN)))
    
    placeholders["arousal_progress"].progress(normalized_progress)
    placeholders["arousal_metric"].metric(label="Arousal Index (Log Ratio)", value=f"{arousal_value:+.2f}")
    placeholders["viability_metric"].metric(label="Viability Band", value=f"[{viability_band[0]:.2f}, {viability_band[1]:.2f}]")

    with placeholders["status_container"]:
        if artifact_detected: st.warning("Artifact Detected!")
        elif in_range: st.success("In Range")
        else: st.error("Out of Range")
    
    

