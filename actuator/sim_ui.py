import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def create_viability_plot(arousal_value, viability_band):
    
    if not viability_band:
        return plt.figure(figsize=(4, 4)) 

    fig, ax = plt.subplots(figsize=(4, 4))

    center_arousal = np.mean(viability_band)
    band_radius = (viability_band[1] - viability_band[0]) / 2.0
    
    if band_radius <= 0:
        band_radius = 0.01 
    
    distance_from_center = arousal_value - center_arousal
    angle = np.random.rand() * 2 * np.pi
    dot_x = distance_from_center * np.cos(angle)
    dot_y = distance_from_center * np.sin(angle)

    
    circle = plt.Circle((0, 0), band_radius, color='lightblue', alpha=0.8, label='Viability Band')
    ax.add_artist(circle)
    dot_color = 'green' if viability_band[0] <= arousal_value <= viability_band[1] else 'red'
    ax.plot(dot_x, dot_y, 'o', color=dot_color, markersize=15, label='Current Arousal')

    
    plot_limit = max(band_radius * 1.5, abs(distance_from_center) * 1.1, 0.1)
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    return fig

#sidebar controls and butons for start/stop
def render_sim():
    st.sidebar.title("Simulation Controls")

    
    st.session_state.setdefault("state_name", "Calm")
    st.session_state.setdefault("target_arousal", 0.75)
    st.session_state.setdefault("natural_flux", 0.10)
    st.session_state.setdefault("latency", 5)
    st.session_state.setdefault("sensor_sampling_rate", 20)
    st.session_state.setdefault("gain", 0.10)
    st.session_state.setdefault("noise_level", 0.01)
    st.session_state.setdefault("feedback_on", True)

    
    state_name = st.sidebar.selectbox(
        "Base State",
        ('Calm', 'Focused', 'Stressed'),
        key="state_name"
    )
    target_arousal = st.sidebar.slider(
        "Target Arousal",
        0.0, 1.0,
        key="target_arousal"
    )
    
    natural_flux = st.sidebar.slider(
        "Viability Band Width",
        0.01, 0.5,
        value=st.session_state["natural_flux"],
        step=0.01,
        key="natural_flux"
    )
    st.sidebar.divider()
    
    st.sidebar.subheader("System Parameters")
    latency = st.sidebar.slider(
        "Latency ", 0, 40,
        value=st.session_state["latency"],
        step=1,
        key="latency"
    )
    sensor_sampling_rate = st.sidebar.slider(
        "Sensor_sampling_Rate (Hz)", 5, 50,
        value=st.session_state["sensor_sampling_rate"],
        step=1,
        key="sensor_sampling_rate"
    )
    gain = st.sidebar.slider(
        "Gain", 0.0, 1.0,
        value=st.session_state["gain"],
        step=0.01,
        key="gain"
    )
    noise_level = st.sidebar.slider(
        "noise_level", 0.0, 0.1,
        value=st.session_state["noise_level"],
        step=0.01,
        format="%.3f",
        key="noise_level"
    )
    feedback_on = st.sidebar.checkbox(
        "FeedBack", value=st.session_state["feedback_on"], key="feedback_on"
    )

    st.sidebar.divider()

    
    st.sidebar.subheader("Manual Configs")
    spike_up = st.sidebar.button("Spike Up (Arousal ++)", key="spike_up")
    spike_down = st.sidebar.button("Spike Down (Arousal --)", key="spike_down")

    st.sidebar.divider()

    
    col1, col2 = st.columns([1, 1])
    with col1:
        start_button = st.button("Start Simulation", key="start_button")
    with col2:
        stop_button = st.button("Stop Simulation", key="stop_button")

    return {
        "state_name": state_name,
        "target_arousal": target_arousal,
        "natural_flux": natural_flux,
        "latency": latency,
        "sensor_sampling_rate": sensor_sampling_rate,
        "gain": gain,
        "noise_level": noise_level,
        "feedback_on": feedback_on,
        "spike_up": spike_up,
        "spike_down": spike_down,
        "start_button": start_button,
        "stop_button": stop_button
    }


#graph + arousal + vb
def render_sim_dashboard():
    live_area = st.container()
    with live_area:
        st.subheader("Live Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            plot_placeholder = st.empty()
        with col2:
            arousal_metric = st.metric(label="Arousal Index", value="0.00")
            viability_metric = st.metric(label="Viability Band", value="[0.00, 0.00]")
            status_container = st.empty()
        history_chart = st.empty()

    return {
        "live_area": live_area,
        "plot_placeholder": plot_placeholder,
        "arousal_metric": arousal_metric,
        "viability_metric": viability_metric,
        "status_container": status_container,
        "history_chart": history_chart
    }

#plot + updater
def update_dashboard(placeholders, arousal, viability_band, history):
    placeholders["arousal_metric"].metric("Arousal Index", f"{arousal:.2f}")
    placeholders["viability_metric"].metric("Viability Band", f"[{viability_band[0]:.2f}, {viability_band[1]:.2f}]")
    in_range = viability_band[0] <= arousal <= viability_band[1]
    with placeholders["status_container"]:
        st.success("In Range") if in_range else st.error("Out of Range")

    fig = create_viability_plot(arousal, viability_band)
    placeholders['plot_placeholder'].pyplot(fig)
    plt.close(fig)

    
    chart_data = history.rename(columns={"arousal": "Arousal", "lower_band": "Lower Band", "upper_band": "Upper Band"})
    placeholders["history_chart"].line_chart(chart_data[['Arousal', 'Lower Band', 'Upper Band']])



def render_sim_analysis(data, target_arousal):
    st.subheader("Post-Simulation Analysis")

    
    data['Position_Error'] = data['arousal'] - target_arousal
    data['Velocity'] = data['arousal'].diff().fillna(0)
    data['Velocity_Error'] = data['Velocity']

    
    cost = np.sum(data['Position_Error']**2) 

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cost (Sum of Squared Error)", f"{cost:.2f}")

    
    st.write("**Position Error (Arousal - Target)**")
    st.line_chart(data['Position_Error'])

    st.write("**Velocity Error (Rate of Change of Arousal)**")
    st.line_chart(data['Velocity_Error'])

    st.write("**Position Error Distribution**")
    fig, ax = plt.subplots()
    ax.hist(data['Position_Error'], bins=30, alpha=0.7, label='Error Distribution')
    ax.axvline(0, color='red', linestyle='--', label='Target')
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)