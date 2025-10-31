import matplotlib
matplotlib.use('TkAgg')

import time
import numpy as np
import matplotlib.pyplot as plt
from brainflow.data_filter import DataFilter, FilterTypes
from queue import Empty

def run_plot(data_queue, sampling_rate, channel_names, window_seconds):
    buffer_size = int(window_seconds * sampling_rate)
    data_buffers = np.zeros((len(channel_names), buffer_size))
 
    plt.ion()
    fig, axs = plt.subplots(len(channel_names), 1, figsize=(12, 9), sharex=True)
    if len(channel_names) == 1:
        axs = [axs]
        
    fig.suptitle('Live EEG Signals', fontsize=16)

    time_axis = np.arange(-window_seconds, 0, 1.0/sampling_rate)
    if len(time_axis) > buffer_size: time_axis = time_axis[:buffer_size]
    
    lines = [ax.plot(time_axis, np.zeros(buffer_size))[0] for ax in axs]
    
    for i, ax in enumerate(axs):
        ax.set_title(channel_names[i])
        ax.set_ylim(-150, 150)
        ax.grid(True)
    
    axs[-1].set_xlabel(f'Time Window ({window_seconds}s)')
    fig.text(0.06, 0.5, 'Voltage (uV)', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.08, 0, 1, 0.96])

    
    while True:
        try:
            new_eeg_data = data_queue.get_nowait()
            
            if new_eeg_data.shape[1] > 0:
                num_new_samples = new_eeg_data.shape[1]

                if num_new_samples > buffer_size:
                    new_eeg_data = new_eeg_data[:, -buffer_size:]
                    num_new_samples = buffer_size

                data_buffers = np.roll(data_buffers, -num_new_samples, axis=1)
                data_buffers[:, -num_new_samples:] = new_eeg_data
                          
                for i, line in enumerate(lines):
                    filtered_channel = data_buffers[i].copy()
                    DataFilter.perform_bandpass(filtered_channel, sampling_rate, 1.0, 45.0, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                    DataFilter.perform_bandstop(filtered_channel, sampling_rate, 48.0, 52.0, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                    line.set_ydata(filtered_channel)
   
                for i, ax in enumerate(axs):
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)

                fig.canvas.draw()
                fig.canvas.flush_events()

        except Empty:
            time.sleep(0.05)
        
        except Exception:
            break
        