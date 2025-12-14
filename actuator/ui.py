import streamlit as st
from brainflow.board_shim import BoardShim, BoardIds
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as patches



#main plot
def create_viability_plot(arousal_value, viability_band):
    fig, ax = plt.subplots(figsize=(8, 2))
    bar_min, bar_max = viability_band[0] - 1.0, viability_band[1] + 1.0
    bar_range = bar_max - bar_min
    
    #bg
    ax.add_patch(patches.Rectangle((bar_min, 0), bar_range, 1, facecolor='gray'))
    
    #viability band
    band_start = viability_band[0]
    band_width = viability_band[1] - viability_band[0]
    ax.add_patch(patches.Rectangle((band_start, 0), band_width, 1, facecolor='deepskyblue'))
    
    #red line
    ax.axvline(x=arousal_value, color='red', linewidth=3)
    ax.set_xlim(bar_min, bar_max)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig


#returns placeholders 
def render_dashboard():    
    live_area = st.container()
    
    with live_area:
        st.subheader("Live Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_placeholder = st.empty()
            arousal_metric = st.metric(label="Arousal Index (Log Ratio)", value="0.00")
            viability_metric = st.metric(label="Viability Band", value="[0.00, 0.00]")
        
        with col2:
            status_container = st.empty()
            st.divider()
            st.subheader("System Metrics")
            st.write("Session Duration")
            session_time_metric = st.empty()
            st.write("Samples Collected")
            samples_metric = st.empty()
            st.write("Artifact Rate")
            artifact_rate_metric = st.empty()
        
        st.divider()
        history_chart = st.empty()

    return {

        "live_area": live_area,
        "plot_placeholder": plot_placeholder,
        "arousal_metric": arousal_metric,
        "viability_metric": viability_metric,
        "status_container": status_container,
        "history_chart": history_chart,
        "session_time_metric": session_time_metric,
        "samples_metric": samples_metric,
        "artifact_rate_metric": artifact_rate_metric,
    }

def update_main_dashboard(placeholders, arousal, viability_band, in_range, artifact_detected, history_df=None, session_stats=None):
    arousal_value = arousal if arousal is not None else 0.0
    
    # visual sys plot
    fig = create_viability_plot(arousal_value, viability_band)
    placeholders['plot_placeholder'].pyplot(fig)
    plt.close(fig)
    
    # update metrics
    placeholders["arousal_metric"].metric(label="Arousal Index (Log Ratio)", value=f"{arousal_value:+.2f}")
    placeholders["viability_metric"].metric(label="Viability Band", value=f"[{viability_band[0]:.2f}, {viability_band[1]:.2f}]")
    
    # update sys status
    with placeholders["status_container"]:
        if artifact_detected:
            st.error("STATUS: ARTIFACT DETECTED")
        elif in_range:
            st.success("STATUS: IN RANGE")
        else:
            st.warning("STATUS: OUT OF RANGE")
    
    #update session stats 
    if session_stats:
        placeholders["session_time_metric"].metric("", f"{session_stats.get('duration', 0):.1f}s")
        placeholders["samples_metric"].metric("", f"{session_stats.get('total_samples', 0)}")
        placeholders["artifact_rate_metric"].metric("", f"{session_stats.get('artifact_rate', 0):.1f}%")
    
    #update history
    if history_df is not None and not history_df.empty:
        chart_data = history_df.rename(columns={
            "arousal": "Arousal",
            "lower_band": "Lower Band",
            "upper_band": "Upper Band"
        })
        placeholders["history_chart"].line_chart(chart_data[['Arousal', 'Lower Band', 'Upper Band']])



# GRAPHS
def render_post_session_analysis(history_df, viability_band, sampling_rate=10):
    if history_df.empty:
        st.info("no data :(")
        return
    
    st.subheader("Post-Session Analysis")
    
    band_center = np.mean(viability_band)
    
    history_df['Position_Error'] = history_df['arousal'] - band_center
    history_df['Velocity'] = history_df['arousal'].diff().fillna(0)
    history_df['Velocity_Error'] = history_df['Velocity']
    
    # energy cost
    cost = np.sum(history_df['Position_Error']**2)
    
    # calc time to goal (3s in band)
    samples_needed = int(3 * sampling_rate)
    time_to_goal = None
    
    if 'in_range' in history_df.columns:
        in_range_series = history_df['in_range'].values
        consecutive_count = 0
        for i, in_range in enumerate(in_range_series):
            if in_range:
                consecutive_count += 1
                if consecutive_count >= samples_needed:
                    time_to_goal = (i + 1) / sampling_rate
                    break
            else:
                consecutive_count = 0
    
    # dl CSV btn
    csv_data = history_df.to_csv(index=True)
    st.download_button(
        label="Download CSV :)",
        data=csv_data,
        file_name=f"session_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cost", f"{cost:.2f}")
    with col2:
        if time_to_goal is not None:
            st.metric("Time to Reach Goal", f"{time_to_goal:.2f} s")
        else:
            st.metric("Time to Reach Goal", "Not reached")
    with col3:
        in_range_pct = (history_df['in_range'].sum() / len(history_df) * 100) if 'in_range' in history_df else 0
        st.metric("Time In Range", f"{in_range_pct:.1f}%")
    with col4:
        artifact_pct = (history_df['artifact'].sum() / len(history_df) * 100) if 'artifact' in history_df else 0
        st.metric("Artifact Rate", f"{artifact_pct:.1f}%")
    
    st.divider()
    
    # Position Error graph
    st.write("**Position Error**")
    st.line_chart(history_df['Position_Error'])
    
    # Velocity Error graph
    st.write("**Velocity Error**")
    st.line_chart(history_df['Velocity_Error'])
    
    # Position Error distr
    st.write("**Position Error Distribution**")
    fig, ax = plt.subplots()
    ax.hist(history_df['Position_Error'], bins=30, alpha=0.7, label='Error Distribution')
    ax.axvline(0, color='red', linestyle='--', label='Target')
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    st.divider()
    
    # phase portrait
    st.write("### Phase Portrait")
    
    phase_data = history_df[['Position_Error', 'Velocity_Error']].copy()
    phase_data['Time_Step'] = range(len(phase_data))
    
    chart = alt.Chart(phase_data).mark_point(opacity=0.5, size=10).encode(
        x=alt.X('Position_Error:Q', title='Position Error'),
        y=alt.Y('Velocity_Error:Q', title='Velocity Error'),
        tooltip=['Time_Step:Q', 'Position_Error:Q', 'Velocity_Error:Q']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)