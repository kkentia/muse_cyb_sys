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


#----------------------------------REAL MODE----------------------------------------------------------------------------

def run_real_mode():
    if 'processor' not in st.session_state: st.session_state.processor = Processor()
    if 'controller' not in st.session_state: st.session_state.controller = Controller()

    processor = st.session_state.processor
    controller = st.session_state.controller
    stream = st.session_state.stream
    
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
    num_channels = len(BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value))
    samples_for_processor = sampling_rate * processor.eeg_window_size

    # memory buffer: stores most recent eeg samples continuously
    if 'main_buffer' not in st.session_state:
        st.session_state.main_buffer = np.empty((num_channels, 0))
    

    st.sidebar.title("Session Control")
    if st.sidebar.button("Stop", key="stop_session_button"):
        stream.release()
        st.session_state.clear()
        st.success("Session stopped. Select a mode to begin a new session.")
        st.rerun()

    
    if not processor.is_calibrated:
            st.title("Step 1: Calibration"); st.info("Goal: relax until the 'Live Variance' is low and stable.")
            st.sidebar.title("Tuning Controls")
            motion_threshold = st.sidebar.slider("Artifact Threshold", 500, 80000, 10000, 500, key="artifact_threshold_slider")
            col1, col2 = st.columns(2)
            with col1: st.write("Progress:"); progress_text = st.empty(); progress_bar = st.progress(0)
            with col2: st.write("Signal Quality:"); variance_text = st.metric("Live Variance", "waiting...")
            if st.button("Start Calibration", key="start_calibration_button"):
                processor.motion_threshold = motion_threshold; st.session_state.baseline_arousal_values = []
                target_samples = 80
                
                sampling_rate = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
                num_channels = len(BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value))
                samples_needed = sampling_rate * processor.eeg_window_size
                
                calibration_buffer = np.empty((num_channels, 0))

                while len(st.session_state.baseline_arousal_values) < target_samples:
                    eeg_data, _ = stream.get_data()
                    
                    if 'plot_queue' in st.session_state and eeg_data.shape[1] > 0:
                        try: st.session_state.plot_queue.put_nowait(eeg_data)
                        except Full: pass
                    
                    if not eeg_data.any(): 
                        time.sleep(0.05)
                        continue
                    
                    
                    # CALIBRATION
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
                
                processor.calibrate(st.session_state.baseline_arousal_values) # -> sets viability band
                st.success("Calibration complete!"); time.sleep(1); st.rerun() 
            return


    if 'ui_placeholders' not in st.session_state:
        st.session_state.ui_placeholders = render_dashboard()

    # main loop
    while True:
        eeg_data, _ = stream.get_data()
        
        if 'plot_queue' in st.session_state and eeg_data.shape[1] > 0:
            try: st.session_state.plot_queue.put_nowait(eeg_data)
            except Full: pass
        
        if not eeg_data.any(): 
            time.sleep(0.02)
            continue


        st.session_state.main_buffer = np.concatenate((st.session_state.main_buffer, eeg_data), axis=1)
    
        if st.session_state.main_buffer.shape[1] >= samples_for_processor:
            
            processing_data = st.session_state.main_buffer[:, -samples_for_processor:]

            arousal, artifact_detected, _ = processor.process_eeg(processing_data.copy())
            in_range, last_good_arousal = controller.update_state(arousal, processor.viability_band, artifact_detected)
            
            update_main_dashboard(
                st.session_state.ui_placeholders, 
                last_good_arousal, 
                processor.viability_band, 
                in_range, 
                artifact_detected
            ) 
            
            max_buffer_size = sampling_rate * 5     # 5 seconds
            if st.session_state.main_buffer.shape[1] > max_buffer_size:
                st.session_state.main_buffer = st.session_state.main_buffer[:, -max_buffer_size:]
        
        time.sleep(0.02)


#--------------------------SIMULATION MODE----------------------------------------------------------------------------------

def run_simulation_mode():
    st.title("System Viability Simulation")

    
    if 'sim_is_running' not in st.session_state:
        st.session_state.sim_is_running = False
    if 'sim_stream' not in st.session_state:
        st.session_state.sim_stream = SimulatedStream()
    if 'sim_history' not in st.session_state:
        st.session_state.sim_history = pd.DataFrame()

    sim_stream = st.session_state.sim_stream
    controls = render_sim()
    live_placeholders = render_sim_dashboard()


    if controls["start_button"]:
        st.session_state.sim_is_running = True
        st.session_state.sim_history = pd.DataFrame(columns=["arousal", "lower_band", "upper_band"])
        sim_stream.reset(controls["state_name"])
        st.rerun() 

    if controls["stop_button"]:
        st.session_state.sim_is_running = False
        st.rerun() 
    
    if controls["spike_up"]:
        sim_stream.apply_spike(0.2)
        if not st.session_state.sim_is_running: st.rerun()

    if controls["spike_down"]:
        sim_stream.apply_spike(-0.2)
        if not st.session_state.sim_is_running: st.rerun()

    
    if st.session_state.sim_is_running:
        st.session_state.get('post_analysis_container', st.empty()).empty()
        while st.session_state.sim_is_running:
            current_controls = {key: st.session_state[key] for key in controls}
            
            sim_stream.set_state(
                current_controls["state_name"],
                current_controls["target_arousal"],
                current_controls["natural_flux"]
            )

            arousal, viability_band = sim_stream.get_arousal_value(
                current_controls["state_name"],
                current_controls["noise_level"],
                current_controls["gain"],
                current_controls["latency"],
                current_controls["feedback_on"]
            )

            new_entry = pd.DataFrame([{"arousal": arousal, "lower_band": viability_band[0], "upper_band": viability_band[1]}])
            st.session_state.sim_history = pd.concat([st.session_state.sim_history, new_entry], ignore_index=True)
            
            if len(st.session_state.sim_history) > 200:
                 st.session_state.sim_history = st.session_state.sim_history.iloc[-200:]

            update_dashboard(live_placeholders, arousal, viability_band, st.session_state.sim_history)

            time.sleep(1 / current_controls["sensor_sampling_rate"])
            
            
            if not st.session_state.sim_is_running:
                break
    
    elif not st.session_state.sim_history.empty:
        live_placeholders["live_area"].empty()
        
        post_analysis_container = st.container()
        st.session_state.post_analysis_container = post_analysis_container
        with post_analysis_container:
            st.info("Simulation stopped. Showing analysis of the collected data.")
            render_sim_analysis(st.session_state.sim_history, st.session_state.target_arousal)



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
            st.info("Starting live plot window")
            board_id = BoardIds.MUSE_2_BOARD.value
            
            plot_queue = mp.Queue()
            st.session_state.plot_queue = plot_queue
            
            plot_process = mp.Process(
                target=run_plot, 
                args=(plot_queue, BoardShim.get_sampling_rate(board_id),BoardShim.get_eeg_names(board_id),5 )
            )
            plot_process.start()
            st.session_state.plot_process = plot_process
            
            time.sleep(1)

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