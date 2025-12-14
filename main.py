from queue import Full
import streamlit as st
import time
import pandas as pd
import numpy as np


from streams.simulated_stream import SimulatedStream
from streams.muse_stream import MuseStream
from processing.processor import Processor
from controller.logic import Controller
from actuator.ui import render_dashboard, update_main_dashboard
from brainflow.board_shim import BoardShim, BoardIds
import multiprocessing as mp 
from plot_stream import run_plot 

from actuator.sim_ui import render_sim, render_sim_dashboard, update_dashboard, render_sim_analysis
from actuator.ui import render_post_session_analysis


#----------------------------------REAL MODE----------------------------------------------------------------------------

def run_real_mode():
    if 'processor' not in st.session_state: st.session_state.processor = Processor()
    if 'controller' not in st.session_state: st.session_state.controller = Controller()
    
    # Initialize history tracking
    if 'real_history' not in st.session_state:
        st.session_state.real_history = pd.DataFrame(columns=["arousal", "lower_band", "upper_band", "in_range", "artifact"])
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = time.time()
    if 'total_samples' not in st.session_state:
        st.session_state.total_samples = 0
    if 'artifact_count' not in st.session_state:
        st.session_state.artifact_count = 0
    if 'session_stopped' not in st.session_state:
        st.session_state.session_stopped = False

    processor = st.session_state.processor
    controller = st.session_state.controller
    stream = st.session_state.stream
    
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
    num_channels = len(BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value))
    samples_for_processor = sampling_rate * processor.eeg_window_size

    # memory buffer: stores most recent eeg samples continuously
    if 'main_buffer' not in st.session_state:
        st.session_state.main_buffer = np.empty((num_channels, 0))
    
    
    # ----------------------------------------- UI STUFF
    # sidebar stop btn
    st.sidebar.title("Session Control")
    if st.sidebar.button("Stop Session", key="stop_real_session"):
        st.session_state.session_stopped = True
        st.rerun()
    
    # sidebar restart btn -> reset
    if st.session_state.get('session_stopped', False):
        if st.sidebar.button("Start New Session", key="restart_session"):
            st.session_state.real_history = pd.DataFrame(columns=["arousal", "lower_band", "upper_band", "in_range", "artifact"])
            st.session_state.session_start_time = time.time()
            st.session_state.total_samples = 0
            st.session_state.artifact_count = 0
            st.session_state.session_stopped = False
            if 'ui_placeholders' in st.session_state:
                del st.session_state.ui_placeholders
            st.rerun()
    
    # viability band IQR adjustement
    if processor.is_calibrated and not st.session_state.get('session_stopped', False):
        st.sidebar.divider()
        
        if 'original_viability_band' not in st.session_state:
            st.session_state.original_viability_band = processor.viability_band.copy()
        
        original_band = st.session_state.original_viability_band
        band_center = np.mean(original_band)
        original_width = original_band[1] - original_band[0]
        
        width_multiplier = st.sidebar.slider(
            "Band Width",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            format="%.1fx",
            key="band_width_multiplier",
        )
        
        new_width = original_width * width_multiplier
        new_lower = band_center - (new_width / 2)
        new_upper = band_center + (new_width / 2)
        
        processor.viability_band = [new_lower, new_upper]
        
    
    # SESSION ANALYSIS
    if st.session_state.session_stopped:
        st.title("Session Stopped")
        if not st.session_state.real_history.empty:
            render_post_session_analysis(st.session_state.real_history, processor.viability_band)
        else:
            st.info("No data was collected during this session.")
        return
    
    # CALIBRATION 
    if not processor.is_calibrated:
        st.title("Step 1: Calibration")
        st.info("Goal: relax until the 'Live Variance' is low and stable.")
        st.sidebar.title("Tuning Controls")
        motion_threshold = st.sidebar.slider("Artifact Threshold", 500, 80000, 10000, 500, key="artifact_threshold_slider")
        col1, col2 = st.columns(2)
        with col1: 
            st.write("Progress:")
            progress_text = st.empty()
            progress_bar = st.progress(0)
        with col2: 
            st.write("Signal Quality:")
            variance_text = st.metric("Live Variance", "waiting...")
        
        if st.button("Start Calibration", key="start_calibration_button"):
            processor.motion_threshold = motion_threshold
            st.session_state.baseline_arousal_values = []
            target_samples = 80
            
            samples_needed = sampling_rate * processor.eeg_window_size
            calibration_buffer = np.empty((num_channels, 0))

            while len(st.session_state.baseline_arousal_values) < target_samples:
                eeg_data = stream.get_data()
                
                if 'plot_queue' in st.session_state and eeg_data.shape[1] > 0:
                    try: st.session_state.plot_queue.put_nowait(eeg_data)
                    except Full: pass
                
                if not eeg_data.any(): 
                    time.sleep(0.05)
                    continue
                
                calibration_buffer = np.concatenate((calibration_buffer, eeg_data), axis=1)

                if calibration_buffer.shape[1] >= samples_needed:
                    data_to_process = calibration_buffer[:, -samples_needed:]
                    arousal, artifact, variance = processor.process_eeg(data_to_process.copy())
                    variance_text.metric("Live Variance", f"{variance:,.0f}")
                    
                    if arousal is not None and not artifact:
                        st.session_state.baseline_arousal_values.append(arousal)
                    
                    progress = len(st.session_state.baseline_arousal_values) / target_samples
                    progress_text.text(f"Collected {len(st.session_state.baseline_arousal_values)}/{target_samples} clean samples...")
                    progress_bar.progress(progress)

                    step_size_seconds = 0.1
                    samples_to_remove = int(sampling_rate * step_size_seconds)
                    calibration_buffer = calibration_buffer[:, samples_to_remove:]
            
            processor.calibrate(st.session_state.baseline_arousal_values) # -> sets viability band around eeg data
            st.success("Calibration complete!")
            time.sleep(1)
            st.rerun()
        return

    # MAIN SESSION 
    if 'ui_placeholders' not in st.session_state:
        st.session_state.ui_placeholders = render_dashboard()

    # MAIN LOOP
    while True:
        eeg_data = stream.get_data()
        
        if 'plot_queue' in st.session_state and eeg_data.shape[1] > 0:
            try: st.session_state.plot_queue.put_nowait(eeg_data)
            except Full: pass
        
        if not eeg_data.any(): 
            time.sleep(0.02)
            continue

        st.session_state.main_buffer = np.concatenate((st.session_state.main_buffer, eeg_data), axis=1)

        if st.session_state.main_buffer.shape[1] >= samples_for_processor:
            processing_data = st.session_state.main_buffer[:, -samples_for_processor:]

            # process EEG data ONCE
            arousal, artifact_detected, _ = processor.process_eeg(processing_data.copy())
            in_range, last_good_arousal = controller.update_state(arousal, processor.viability_band, artifact_detected)
            
            # track history
            st.session_state.total_samples += 1
            if artifact_detected:
                st.session_state.artifact_count += 1

            new_entry = pd.DataFrame([{
                "arousal": last_good_arousal,
                "lower_band": processor.viability_band[0],
                "upper_band": processor.viability_band[1],
                "in_range": in_range,
                "artifact": artifact_detected
            }])
            st.session_state.real_history = pd.concat([st.session_state.real_history, new_entry], ignore_index=True)
            if len(st.session_state.real_history) > 200:
                st.session_state.real_history = st.session_state.real_history.iloc[-200:]

            # calc session stats
            session_duration = time.time() - st.session_state.session_start_time
            artifact_rate = (st.session_state.artifact_count / st.session_state.total_samples * 100) if st.session_state.total_samples > 0 else 0
            session_stats = {
                'duration': session_duration,
                'total_samples': st.session_state.total_samples,
                'artifact_rate': artifact_rate
            }

            update_main_dashboard(
                st.session_state.ui_placeholders, 
                last_good_arousal, 
                processor.viability_band, 
                in_range, 
                artifact_detected,
                st.session_state.real_history,
                session_stats
            )
            
            max_buffer_size = sampling_rate * 5 # 5seconds
            if st.session_state.main_buffer.shape[1] > max_buffer_size:
                st.session_state.main_buffer = st.session_state.main_buffer[:, -max_buffer_size:]
        
        time.sleep(0.02)
        
        
        
#--------------------------SIMULATION MODE----------------------------------------------------------------------------------

def run_simulation_mode():
    st.title("System Viability Simulation")

    if 'sim_is_running' not in st.session_state: st.session_state.sim_is_running = False
    if 'sim_stream' not in st.session_state: st.session_state.sim_stream = SimulatedStream()
    if 'sim_history' not in st.session_state: st.session_state.sim_history = pd.DataFrame()
    
    def set_scenario(scenario_name):
        st.session_state.sim_is_running = True
        st.session_state.sim_history = pd.DataFrame(columns=["arousal", "lower_band", "upper_band", "fatigue", "energy", "energy_spent", "in_band"])
        
        if scenario_name == "caffeine":
            st.session_state.gain = 0.5; st.session_state.latency = 2; st.session_state.noise_level = 0.02; st.session_state.environmental_threat = 0.0; st.session_state.effort_amplification = 5.0
        elif scenario_name == "drowsy":
            st.session_state.state_name = "Focused"; st.session_state.gain = 0.05; st.session_state.latency = 30; st.session_state.noise_level = 0.01; st.session_state.environmental_threat = 0.0; st.session_state.effort_amplification = 3.0
        elif scenario_name == "exam":
            st.session_state.state_name = "Focused"; st.session_state.target_arousal = 0.6; st.session_state.gain = 0.15; st.session_state.environmental_threat = 0.4; st.session_state.effort_amplification = 5.0
        
        st.session_state.sim_stream.reset(st.session_state.state_name)

    sim_stream = st.session_state.sim_stream
    controls = render_sim(
        on_caffeine_click=lambda: set_scenario("caffeine"),
        on_drowsy_click=lambda: set_scenario("drowsy"),
        on_exam_click=lambda: set_scenario("exam")
    )
    live_placeholders = render_sim_dashboard()

    state_intervals = {}
    FIXED_FLUX_FOR_DISPLAY = 0.05 
    for state, values in sim_stream.states.items():
        initial = values['initial_arousal']
        state_intervals[state] = [initial - FIXED_FLUX_FOR_DISPLAY, initial + FIXED_FLUX_FOR_DISPLAY]

    if controls["start_button"]:
        st.session_state.sim_is_running = True
        st.session_state.sim_history = pd.DataFrame(columns=["arousal", "lower_band", "upper_band", "fatigue", "energy", "energy_spent", "in_band"])
        sim_stream.reset(controls["state_name"])
        # clear
        if 'post_analysis_container' in st.session_state:
            st.session_state.post_analysis_container.empty()
            del st.session_state.post_analysis_container
        st.rerun()

    if controls["stop_button"]: st.session_state.sim_is_running = False; st.rerun()
    if controls["spike_up"]: sim_stream.apply_spike(0.2);
    if controls["spike_down"]: sim_stream.apply_spike(-0.2);

    if controls["auto_tune_button"]:
        sim_stream.start_auto_tuning()
        st.info("Auto-Tuning process started...")
        
    if st.session_state.sim_is_running:
        st.session_state.get('post_analysis_container', st.empty()).empty()
        while st.session_state.sim_is_running:
            control_keys = [
                "state_name", "target_arousal", "natural_flux", "latency", 
                "sensor_sampling_rate", "kp", "ki", "kd", "noise_level", 
                "feedback_on", "environmental_threat", "effort_amplification", 
                "controller_type"
            ]
            current_controls = {key: st.session_state.get(key) for key in control_keys}
        
            # pass new PID gains & controller type
            arousal, viability_band, fatigue, is_burnt_out, energy, pid_gains, energy_spent = sim_stream.get_arousal_value(
                current_controls["state_name"], current_controls["target_arousal"],
                current_controls["natural_flux"], current_controls["noise_level"], 
                current_controls["kp"], current_controls["ki"], current_controls["kd"],
                current_controls["latency"], current_controls["feedback_on"],
                current_controls["environmental_threat"], current_controls["effort_amplification"],
                current_controls["controller_type"]  

            )

            # check if in viability band
            in_band = viability_band[0] <= arousal <= viability_band[1]

            new_entry = pd.DataFrame([{
                "arousal": arousal, 
                "lower_band": viability_band[0], 
                "upper_band": viability_band[1], 
                "fatigue": fatigue, 
                "energy": energy,
                "energy_spent": energy_spent,
                "in_band": in_band
            }])
            st.session_state.sim_history = pd.concat([st.session_state.sim_history, new_entry], ignore_index=True)
            if len(st.session_state.sim_history) > 200: st.session_state.sim_history = st.session_state.sim_history.iloc[-200:]
            
            update_dashboard(live_placeholders, arousal, viability_band, st.session_state.sim_history, current_controls["noise_level"], fatigue, is_burnt_out, state_intervals, energy, pid_gains)            
            time.sleep(1 / current_controls["sensor_sampling_rate"])
            
            if not st.session_state.sim_is_running: break
    
    elif not st.session_state.sim_history.empty:
        live_placeholders["live_area"].empty()
        post_analysis_container = st.container()
        st.session_state.post_analysis_container = post_analysis_container
        with post_analysis_container:
            st.info("Simulation stopped. Showing analysis of the collected data.")
            sampling_rate = st.session_state.get("sensor_sampling_rate", 20)
            render_sim_analysis(st.session_state.sim_history, st.session_state.target_arousal, sampling_rate)

#----------------------------------------------------------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide")

    if 'mode' not in st.session_state: st.session_state.mode = None

    mode = st.sidebar.selectbox(
        "Select Application Mode",
        ("Select a mode...", "Live EEG", "Simulation Mode"),
        key='mode_selector'
    )

    #checks for mode change -> queue for multiprocessing
    if mode != st.session_state.mode:
        if 'plot_process' in st.session_state and st.session_state.plot_process.is_alive():
            st.session_state.plot_process.terminate()
            st.session_state.plot_process.join() 
            st.session_state.pop('plot_process')
            st.session_state.pop('plot_queue')
        st.session_state.mode = mode
        st.rerun()


    if mode == "Live EEG":
        if 'plot_process' not in st.session_state:
            st.info("Starting live plot window...")
            board_id = BoardIds.MUSE_2_BOARD.value
            
            plot_queue = mp.Queue()
            st.session_state.plot_queue = plot_queue
            
            plot_process = mp.Process(
                target=run_plot, 
                args=(plot_queue, BoardShim.get_sampling_rate(board_id), BoardShim.get_eeg_names(board_id), 5),
                daemon=True  #clean close
            )
            plot_process.start()
            st.session_state.plot_process = plot_process
            
            time.sleep(2)  
            st.success("Plot window started!")

        if 'stream' not in st.session_state:
            try:
                st.session_state.stream = MuseStream(data_queue=st.session_state.plot_queue)
            except Exception as e:
                st.error(f"Failed to connect to Muse: {e}")
                st.stop()
        run_real_mode()

    elif mode == "Simulation Mode":
        run_simulation_mode()

    else:
        st.title("Muse Arousal Stability System")
        st.write("Select a mode from the sidebar.")

if __name__ == "__main__":
    mp.freeze_support() 
    main()