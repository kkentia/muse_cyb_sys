import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import altair as alt
import pandas as pd

def create_viability_plot(arousal_value, viability_band, noise_level):
    fig, ax = plt.subplots(figsize=(8, 2))
    bar_min, bar_max = 0.0, 1.0
    bar_range = bar_max - bar_min
    ax.add_patch(patches.Rectangle((bar_min, 0), bar_range, 1, facecolor='gray'))
    if viability_band and len(viability_band) == 2:
        band_start = viability_band[0]
        band_width = viability_band[1] - viability_band[0]
        ax.add_patch(patches.Rectangle((band_start, 0), band_width, 1, facecolor='deepskyblue'))
    ax.axvline(x=arousal_value, color='red', linewidth=3)
    ax.set_xlim(bar_min, bar_max)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

def render_sim(on_caffeine_click, on_drowsy_click, on_exam_click):
    st.sidebar.title("Simulation Controls")

    st.session_state.setdefault("state_name", "Calm")
    st.session_state.setdefault("target_arousal", 0.75)
    st.session_state.setdefault("natural_flux", 0.10)
    st.session_state.setdefault("latency", 5)
    st.session_state.setdefault("sensor_sampling_rate", 20)
    st.session_state.setdefault("noise_level", 0.005)
    st.session_state.setdefault("feedback_on", False)
    st.session_state.setdefault("environmental_threat", 0.0)
    st.session_state.setdefault("effort_amplification", 4.0)
    st.session_state.setdefault("kp", 0.30)  
    st.session_state.setdefault("ki", 0.05)  
    st.session_state.setdefault("kd", 0.15)  
    st.session_state.setdefault("controller_type", "P Controller")

    state_name = st.sidebar.selectbox("Base State", ('Calm', 'Focused', 'Stressed'), key="state_name")
    target_arousal = st.sidebar.slider("Target Arousal", 0.0, 1.0, key="target_arousal")
    natural_flux = st.sidebar.slider("Viability Band Width", 0.01, 0.5, key="natural_flux")
    st.sidebar.divider()
    
    st.sidebar.subheader("Controller Settings")
    controller_type = st.sidebar.radio(
        "Select Controller Type",
        ("P Controller", "PID Controller"),
        key="controller_type"
    )

    if controller_type == "P Controller":
        kp = st.sidebar.slider("Proportional Gain (Kp)", 0.0, 1.0, key="kp")
        ki = st.session_state.get("ki", 0.0)
        kd = st.session_state.get("kd", 0.0)
        st.sidebar.button("Auto-Tune PID Gains", use_container_width=True, disabled=True)
        auto_tune_button = False
    else: # PID controller
        kp = st.sidebar.slider("Proportional Gain (Kp)", 0.0, 1.0, key="kp")
        ki = st.sidebar.slider("Integral Gain (Ki)", 0.0, 0.1, format="%.4f", key="ki")
        kd = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 1.0, key="kd")
        auto_tune_button = st.sidebar.button("Auto-Tune PID Gains", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("System Parameters")
    latency = st.sidebar.slider("Latency ", 0, 40, key="latency")
    sensor_sampling_rate = st.sidebar.slider("Sensor_sampling_Rate (Hz)", 5, 50, key="sensor_sampling_rate")
    effort_amplification = st.sidebar.slider("Effort Amplification", 1.0, 10.0, key="effort_amplification")
    noise_level = st.sidebar.slider("noise_level", 0.0, 0.1, format="%.3f", key="noise_level")
    feedback_on = st.sidebar.checkbox("FeedBack", key="feedback_on")
    st.sidebar.divider()
    st.sidebar.subheader("Environmental Factors")
    environmental_threat = st.sidebar.slider("Environmental Distraction / Threat", 0.0, 1.0, key="environmental_threat")
    st.sidebar.divider()
    st.sidebar.subheader("Manual Configs")
    spike_up = st.sidebar.button("Spike Up (Arousal ++)", key="spike_up")
    spike_down = st.sidebar.button("Spike Down (Arousal --)", key="spike_down")
    st.sidebar.divider()
    col1, col2 = st.columns([1, 1])
    with col1: start_button = st.button("Start Simulation", key="start_button", use_container_width=True)
    with col2: stop_button = st.button("Stop Simulation", key="stop_button", use_container_width=True)
    st.sidebar.subheader("Run Extreme Trials")
    scenario_caffeine = st.sidebar.button("Caffeine shot", on_click=on_caffeine_click)
    scenario_drowsy = st.sidebar.button("Alcohol", on_click=on_drowsy_click)
    scenario_exam = st.sidebar.button("Stressful Exam", on_click=on_exam_click)

    return {
        "state_name": state_name, "target_arousal": target_arousal,
        "natural_flux": natural_flux, "latency": latency,
        "sensor_sampling_rate": sensor_sampling_rate,
        "noise_level": noise_level, "feedback_on": feedback_on,
        "spike_up": spike_up, "spike_down": spike_down,
        "start_button": start_button, "stop_button": stop_button,
        "environmental_threat": environmental_threat,
        "effort_amplification": effort_amplification,
        "kp": kp, "ki": ki, "kd": kd,
        "auto_tune_button": auto_tune_button,
        "controller_type": controller_type,
    }

def render_sim_dashboard():
    live_area = st.container()
    with live_area:
        st.subheader("Live Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            plot_placeholder = st.empty()
            fatigue_bar_label = st.empty()
            fatigue_bar = st.empty()
            energy_bar_label = st.empty()
            energy_bar = st.empty()
        with col2:
            arousal_metric = st.metric(label="Arousal Index", value="0.00")
            viability_metric = st.metric(label="Viability Band", value="[0.00, 0.00]")
            status_container = st.empty()
            st.divider()
            st.subheader("Active PID Gains")
            st.write("Kp (Proportional)"); kp_bar = st.progress(0.0)
            st.write("Ki (Integral)"); ki_bar = st.progress(0.0)
            st.write("Kd (Derivative)"); kd_bar = st.progress(0.0)
            with st.expander("Show Reference Intervals"):
                calm_interval_metric = st.metric(label="Calm", value="[0.00, 0.00]")
                focused_interval_metric = st.metric(label="Focused", value="[0.00, 0.00]")
                stressed_interval_metric = st.metric(label="Stressed", value="[0.00, 0.00]")
        history_chart = st.empty()

    return {
        "live_area": live_area, "plot_placeholder": plot_placeholder,
        "arousal_metric": arousal_metric, "viability_metric": viability_metric,
        "status_container": status_container, "history_chart": history_chart,
        "fatigue_bar_label": fatigue_bar_label, "fatigue_bar": fatigue_bar,
        "energy_bar_label": energy_bar_label, "energy_bar": energy_bar,
        "calm_interval_metric": calm_interval_metric,
        "focused_interval_metric": focused_interval_metric,
        "stressed_interval_metric": stressed_interval_metric,
        "kp_bar": kp_bar, "ki_bar": ki_bar, "kd_bar": kd_bar,
    }

def update_dashboard(placeholders, arousal, viability_band, history, noise_level, fatigue, is_burnt_out, state_intervals, energy, pid_gains):
    placeholders["arousal_metric"].metric("Arousal Index", f"{arousal:.2f}")
    placeholders["viability_metric"].metric("Viability Band", f"[{viability_band[0]:.2f}, {viability_band[1]:.2f}]")
    with placeholders["status_container"]:
        if is_burnt_out: st.error("STATUS: BURNOUT")
        else:
            in_range = viability_band[0] <= arousal <= viability_band[1]
            if in_range: st.success("STATUS: IN RANGE")
            else: st.warning("STATUS: OUT OF RANGE")

    fig = create_viability_plot(arousal, viability_band, noise_level)
    placeholders['plot_placeholder'].pyplot(fig)
    plt.close(fig)

    placeholders["fatigue_bar_label"].text(f"Fatigue Level: {fatigue:.0%}")
    placeholders["fatigue_bar"].progress(fatigue)
    placeholders["energy_bar_label"].text(f"Energy Level: {energy:.0%}")
    placeholders["energy_bar"].progress(energy)

    kp, ki, kd = pid_gains
    placeholders["kp_bar"].progress(np.clip(kp, 0.0, 1.0))
    placeholders["ki_bar"].progress(np.clip(ki * 10, 0.0, 1.0))
    placeholders["kd_bar"].progress(np.clip(kd, 0.0, 1.0))

    if state_intervals:
        placeholders["calm_interval_metric"].metric("Calm", f"[{state_intervals['Calm'][0]:.2f}, {state_intervals['Calm'][1]:.2f}]")
        placeholders["focused_interval_metric"].metric("Focused", f"[{state_intervals['Focused'][0]:.2f}, {state_intervals['Focused'][1]:.2f}]")
        placeholders["stressed_interval_metric"].metric("Stressed", f"[{state_intervals['Stressed'][0]:.2f}, {state_intervals['Stressed'][1]:.2f}]")
    
    chart_data = history.rename(columns={"arousal": "Arousal", "lower_band": "Lower Band", "upper_band": "Upper Band"})
    placeholders["history_chart"].line_chart(chart_data[['Arousal', 'Lower Band', 'Upper Band']])

def render_sim_analysis(data, target_arousal, sampling_rate=20):
    st.subheader("Post-Simulation Analysis")
    
    # calc errors
    data['Position_Error'] = data['arousal'] - target_arousal
    data['Velocity'] = data['arousal'].diff().fillna(0)
    data['Velocity_Error'] = data['Velocity']
    cost = np.sum(data['Position_Error']**2)
    
    # calc time to goal (3s in band)
    samples_needed = int(3 * sampling_rate)  #3 seconds worth
    time_to_goal = None
    energy_at_goal = None
    
    if 'in_band' in data.columns:
        in_band_series = data['in_band'].values
        consecutive_count = 0
        for i, in_band in enumerate(in_band_series):
            if in_band:
                consecutive_count += 1
                if consecutive_count >= samples_needed:
                    # time is total time from start to NOW (i)
                    time_to_goal = (i + 1) / sampling_rate  
                    if 'energy' in data.columns:
                        energy_at_goal = data['energy'].iloc[i]
                    break
            else:
                #if out_of_range, reset counter
                consecutive_count = 0
    
    # calc tot energy spent (cumulative)
    energy_spent_at_goal = None
    if 'energy_spent' in data.columns:
        if time_to_goal is not None:
            goal_sample_index = int(time_to_goal * sampling_rate) - 1
            energy_spent_at_goal = data['energy_spent'].iloc[:goal_sample_index + 1].sum()
    
    # dl CSV btn
    csv_data = data.to_csv(index=False)
    st.download_button(
        label="Download CSV :)",
        data=csv_data,
        file_name=f"simulation_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.divider()
    
    # metrics
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Cost (SSE)", f"{cost:.2f}")
    
    with col2:
        if time_to_goal is not None:
            st.metric("Time to Reach Goal", f"{time_to_goal:.2f} s")
        else:
            st.metric("Time to Reach Goal", "Not reached")

    #2nd row 
    col3, col4 = st.columns(2)
    
    with col3:
        if energy_at_goal is not None:
            st.metric("Energy Remaining at Goal", f"{energy_at_goal:.2%}")
        else:
            st.metric("Energy Remaining at Goal", "N/A")
    
    with col4:
        if energy_spent_at_goal is not None:
            st.metric("Energy Spent to Reach Goal", f"{energy_spent_at_goal:.2%}")
        else:
            st.metric("Energy Spent to Reach Goal", "N/A")
    
    st.divider()
    
    st.write("**Position Error (Arousal - Target)**")
    st.line_chart(data['Position_Error'])
    st.write("**Velocity Error (Rate of Change of Arousal)**")
    st.line_chart(data['Velocity_Error'])
    st.write("**Position Error Distribution**")
    fig, ax = plt.subplots()
    ax.hist(data['Position_Error'], bins=30, alpha=0.7, label='Error Distribution')
    ax.axvline(0, color='red', linestyle='--', label='Target')
    ax.set_xlabel("Error"); ax.set_ylabel("Frequency"); ax.legend()
    st.pyplot(fig)
    st.divider()
    st.write("### Phase Portrait (System Stability)")
    st.write("This plot shows the system's trajectory. A stable system will spiral into the center (0,0).")
    phase_data = data[['Position_Error', 'Velocity_Error']].copy()
    phase_data['Index'] = range(len(phase_data))
    chart = alt.Chart(phase_data).mark_point(opacity=0.5, size=10).encode(
        x=alt.X('Position_Error:Q', title='Position Error'),
        y=alt.Y('Velocity_Error:Q', title='Velocity Error'),
        tooltip=['Index:Q', 'Position_Error:Q', 'Velocity_Error:Q']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)