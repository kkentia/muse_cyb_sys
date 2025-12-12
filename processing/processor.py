import numpy as np
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes
from brainflow.board_shim import BoardShim, BoardIds

class Processor:
    def __init__(self, eeg_window_size=2):
        self.board_id = BoardIds.MUSE_2_BOARD.value
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.channel_names = BoardShim.get_eeg_names(self.board_id)
        self.eeg_window_size = eeg_window_size
        
        self.posterior_channel_indices = [i for i, name in enumerate(self.channel_names) if name in ['TP9', 'TP10']]
        self.frontal_channel_indices = [i for i, name in enumerate(self.channel_names) if name in ['AF7', 'AF8']]     
        
        self.ema_alpha = 0.1
        self.is_calibrated = False
        self.viability_band = [0.4, 0.6]
        self.motion_threshold = 10000
        
        self.last_raw_eeg = None
        self.last_filtered_eeg = None


    # shape = n_channels x n_samples
    def process_eeg(self, eeg_data):
        if eeg_data.shape[1] < self.sampling_rate * self.eeg_window_size: #check window length seconds
            return None, False, 0.0

        self.last_raw_eeg = eeg_data.copy()

        #remove noise (openBCI code)
        for i in range(len(self.eeg_channels)):
            DataFilter.detrend(eeg_data[i], DetrendOperations.CONSTANT.value)
            DataFilter.remove_environmental_noise(eeg_data[i], self.sampling_rate, NoiseTypes.FIFTY.value)
        self.last_filtered_eeg = eeg_data.copy()

        current_variance = np.var(eeg_data[0])
        if current_variance > self.motion_threshold:
            return None, True, current_variance


        #take log values of PSDs to compress the range & normalize variance
        posterior_bands = DataFilter.get_avg_band_powers(eeg_data, self.posterior_channel_indices, self.sampling_rate, True)
        posterior_alpha_log = posterior_bands[0][2]

        frontal_bands = DataFilter.get_avg_band_powers(eeg_data, self.frontal_channel_indices, self.sampling_rate, True)
        frontal_beta_log = frontal_bands[0][3]
        
        # AROUSAL calc
        arousal_index = posterior_alpha_log - frontal_beta_log
            # alpha: 8-12 Hz -> calm, eyes closed -> low arousal
            # beta: 13-30 Hz -> alert, active thinking -> high arousal
            
            
        # smooth with EMA (exponential moving average)
        if not hasattr(self, 'smoothed_arousal'):
            self.smoothed_arousal = arousal_index
        else:
            self.smoothed_arousal = self.ema_alpha * arousal_index + (1 - self.ema_alpha) * self.smoothed_arousal

        return self.smoothed_arousal, False, current_variance
   
   
    #calc viability band -> derive it from past arousal values
    def calibrate(self, arousal_values):
        print("\n STARTING CALIBRATION CALCULATION")
        if arousal_values: # IQR
            lower_bound = np.percentile(arousal_values, 25)  # Q1
            upper_bound = np.percentile(arousal_values, 75)  # Q3
            
            self.viability_band = [lower_bound, upper_bound]
            self.is_calibrated = True
            
            print(f"CALIBRATION COMPLETE")
            print(f"New Viability Band: [{self.viability_band[0]:.3f}, {self.viability_band[1]:.3f}]\n")