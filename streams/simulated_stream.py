import numpy as np
from collections import deque

#generates arousal signal
class SimulatedStream:
    def __init__(self):
        
        self.states = {
            'Calm': {'target_arousal': 0.75, 'natural_flux': 0.05}, # N_f: how much the arousal can move aournd without being out of range
            'Focused': {'target_arousal': 0.50, 'natural_flux': 0.10},
            'Stressed': {'target_arousal': 0.25, 'natural_flux': 0.15}
        }
        self.current_arousal = self.states['Calm']['target_arousal']
        self.viability_band = [0.70, 0.80] 

        
        self.arousal_history = deque(maxlen=50) # last 50 samples

#state and viability band
    def set_state(self, state_name, target_arousal, natural_flux):
        if state_name in self.states:
            self.states[state_name]['target_arousal'] = target_arousal
            self.states[state_name]['natural_flux'] = natural_flux
            self.viability_band = [target_arousal - natural_flux, target_arousal + natural_flux]
        else:
            print(f"state '{state_name}' not found.")

    def reset(self, state_name):
        self.current_arousal = self.states[state_name]['target_arousal']
        self.arousal_history.clear()

    # perturbation
    def apply_spike(self, spike_magnitude):
        self.current_arousal += spike_magnitude
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))


    def get_arousal_value(self, state_name, noise_level, gain, control_delay, feedback_on):
        self.arousal_history.append(self.current_arousal)

        target = self.states[state_name]['target_arousal']
        reversion_force = 0.0 # for below; 0 = random

        # FEEDBACK: pull back arousal to target
        if feedback_on:
            
            if len(self.arousal_history) > control_delay:
                delayed_arousal = self.arousal_history[-(control_delay + 1)]
                error = target - delayed_arousal
                reversion_force = error * gain 
            else:
                error = target - self.current_arousal
                reversion_force = error * gain

        # + NOISE
        random_noise = (np.random.randn() * noise_level) # gaussian noise ; noise_lvl is how much random noise
        self.current_arousal += reversion_force + random_noise
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))

        return self.current_arousal, self.viability_band