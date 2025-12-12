import numpy as np
from collections import deque

#generates arousal signal
class SimulatedStream:
    def __init__(self):
        self.states = {
            'Calm': {'initial_arousal': 0.25}, 
            'Focused': {'initial_arousal': 0.60},
            'Stressed': {'initial_arousal': 0.85}
        }
        self.current_arousal = self.states['Calm']['initial_arousal']
        self.viability_band = [0.20, 0.30]
        
        self.arousal_history = deque(maxlen=200)
        self.fatigue = 0.0
        self.outer_loop_counter = 0
        self.target_override = 0.0
        self.is_burnt_out = False
        self.burnout_recovery_target = 0.0  # go to 0 arousal whne burnt_out
        self.energy = 1.0
        
        # PID state vars
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.kp = 0.1; self.ki = 0.0; self.kd = 0.0
        
        self.is_tuning = False
        self.tuning_kp = 0.0
        self.tuning_ki = 0.0
        self.tuning_kd = 0.0
        self.tuning_peak_times = []
        self.tuning_peak_values = []
        self.tuning_phase = "increasing_kp"  # phases: increasing_kp, measuring_oscillation, monitoring
        self.tuning_last_error = 0.0
        self.tuning_oscillation_count = 0
        self.tuning_ku = 0.0  # uultimate gain
        self.tuning_tu = 0.0  # oscillation time
        self.tuning_steady_state_counter = 0
        self.tuning_steady_state_error_sum = 0.0
    
    
    
    
    def reset(self, state_name):
        self.current_arousal = self.states[state_name]['initial_arousal']
        self.arousal_history.clear()
        self.fatigue = 0.0; self.outer_loop_counter = 0; self.target_override = 0.0
        self.is_burnt_out = False; self.energy = 1.0
        self.integral_error = 0.0; self.previous_error = 0.0
        self.kp = 0.1; self.ki = 0.0; self.kd = 0.0
        self.is_tuning = False; self.tuning_kp = 0.0; self.tuning_ki = 0.0; self.tuning_kd = 0.0
        self.tuning_peak_times = []; self.tuning_peak_values = []
        self.tuning_phase = "increasing_kp"; self.tuning_last_error = 0.0
        self.tuning_oscillation_count = 0; self.tuning_ku = 0.0; self.tuning_tu = 0.0
        self.tuning_steady_state_counter = 0; self.tuning_steady_state_error_sum = 0.0
        self.burnout_recovery_target = 0.0  # reset burnout target
    
    
    
    #perturbation
    def apply_spike(self, spike_magnitude):
        self.current_arousal += spike_magnitude
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))
        
        
        
        
    
    def start_auto_tuning(self):
        print("AUTO-TUNEing......")
        self.is_tuning = True
        self.tuning_kp = 0.05
        self.tuning_ki = 0.0
        self.tuning_kd = 0.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.tuning_peak_times = []
        self.tuning_peak_values = []
        self.tuning_phase = "increasing_kp"
        self.tuning_last_error = 0.0
        self.tuning_oscillation_count = 0
        self.tuning_ku = 0.0
        self.tuning_tu = 0.0
        self.tuning_steady_state_counter = 0
        self.tuning_steady_state_error_sum = 0.0
    
    
    
    
    
    # MAIN METHOD
    
    def get_arousal_value(self, state_name, manual_target_arousal, natural_flux, noise_level, kp, ki, kd, control_delay, feedback_on, environmental_threat, effort_amplification, controller_type):
        # energy regen
        energy_regeneration_rate = 0.001
        if self.current_arousal < 0.4: energy_regeneration_rate *= 2
        self.energy += energy_regeneration_rate
        self.energy = min(1.0, self.energy)
        
        # fatigue & burnout logic
        self.outer_loop_counter += 1
        if self.outer_loop_counter >= 20:
            self.outer_loop_counter = 0
            if not self.is_burnt_out:
                if self.current_arousal > 0.6: self.fatigue += 0.05
                elif self.current_arousal < 0.4: self.fatigue -= 0.1
            else:
                if self.current_arousal < 0.4:
                    if feedback_on: self.fatigue -= 0.01
                    else: self.fatigue -= 0.05
            self.fatigue = max(0.0, min(1.0, self.fatigue))
            if self.fatigue >= 1.0: self.is_burnt_out = True
            if self.is_burnt_out and self.fatigue <= 0.0: self.is_burnt_out = False
            if self.fatigue > 0.7 and not self.is_burnt_out:
                self.target_override = (0.25 - manual_target_arousal) * 0.1
            else: self.target_override = 0.0
        
        self.arousal_history.append(self.current_arousal)
        effective_target = manual_target_arousal + self.target_override
        target = effective_target
        
        # calculate error (w & w/out delay )
        if len(self.arousal_history) > control_delay:
            error = target - self.arousal_history[-(control_delay + 1)]
        else:
            error = target - self.current_arousal
        
        conscious_effort_force = 0.0
        
        if self.is_tuning and feedback_on:
            # Ziegler-Nichols autotune
            if self.tuning_phase == "increasing_kp":
                #  1) increase kp until we get sustained oscillations
                self.tuning_kp += 0.0005
                conscious_effort_force = error * self.tuning_kp
                
                # 2) detect oscillations by checking if error crosses 0
                if len(self.arousal_history) > 2:
                    error_sign_change = (self.tuning_last_error * error) < 0
                    if error_sign_change and abs(error) > 0.02:  # big oscillation
                        self.tuning_peak_times.append(len(self.arousal_history))
                        self.tuning_peak_values.append(abs(error))
                        self.tuning_oscillation_count += 1
                
                self.tuning_last_error = error
                
                # 3) if theres sustained oscillations -->measurement phase
                if self.tuning_oscillation_count >= 6:
                    # calc oscillation period (avg time between peaks)
                    if len(self.tuning_peak_times) >= 4:
                        periods = [self.tuning_peak_times[i] - self.tuning_peak_times[i-2] 
                                 for i in range(2, len(self.tuning_peak_times))]
                        self.tuning_tu = np.mean(periods) if periods else 20
                        self.tuning_ku = self.tuning_kp
                        
                        # Ziegler-Nichols PID equatioms
                        self.kp = 0.6 * self.tuning_ku
                        self.ki = (2.0 * self.kp) / self.tuning_tu
                        self.kd = (self.kp * self.tuning_tu) / 8.0
                        
                        # safety to extreme values
                        
                        self.kp = min(self.kp, 0.5)  
                        self.ki = min(self.ki, 0.05)  
                        self.kd = min(self.kd, 0.3)  
                        ''' max values
                        self.tuning_kp = self.kp
                        self.tuning_ki = self.ki
                        self.tuning_kd = self.kd
                        '''
                        self.tuning_phase = "done"
                        print(f"Auto-tuning complete! Ku={self.tuning_ku:.4f}, Tu={self.tuning_tu:.1f}")
                        print(f"PID Gains: Kp={self.kp:.4f}, Ki={self.ki:.4f}, Kd={self.kd:.4f}")
                        self.is_tuning = False
                
                #stop if kp too high
                if self.tuning_kp > 1.0:
                    print("Auto-tuning failed: kp went over limit without stable oscillations")
                    self.tuning_kp = 0.3
                    self.kp = 0.3
                    self.ki = 0.01
                    self.kd = 0.05
                    self.tuning_ki = self.ki
                    self.tuning_kd = self.kd
                    self.is_tuning = False
            
            display_kp = self.tuning_kp
            display_ki = self.tuning_ki
            display_kd = self.tuning_kd
        
        elif self.is_burnt_out:
            conscious_effort_force = (self.burnout_recovery_target - self.current_arousal) * 0.05

        
        
        # FEEDBACK: pull back arousal to target
        elif feedback_on:
            base_reversion_force = 0.0  
            
            if controller_type == "P Controller":
                base_reversion_force = error * kp
                self.integral_error = 0.0
                self.previous_error = 0.0
                
            elif controller_type == "PID Controller":
                self.integral_error += error
                self.integral_error = max(-5.0, min(5.0, self.integral_error))
                derivative_error = error - self.previous_error
                self.previous_error = error
                
                base_reversion_force = (kp * error) + (ki * self.integral_error) + (kd * derivative_error) #BASE CONTROLLING FORCE
            
            #amp and lim
            amplifier = 1.0 + (abs(error) * effort_amplification)
            amplified_force = base_reversion_force * amplifier
            energy_limitation_factor = self.energy
            fatigue_degradation_factor = 1.0 - ((self.fatigue - 0.7) / 0.3) if self.fatigue > 0.7 else 1.0
            conscious_effort_force = amplified_force * fatigue_degradation_factor * energy_limitation_factor
        
        else: # if FEEDBACK_ON FALSE
            resting_target = self.states[state_name]['initial_arousal']
            drift_error = resting_target - self.current_arousal
            conscious_effort_force = drift_error * 0.02
            self.integral_error = 0.0
            self.previous_error = 0.0
            if self.is_tuning:
                print("STOPPING auto-tuner")
                self.is_tuning = False
        
        # ENERGY COST
        energy_cost_multiplier = 0.1
        energy_cost = abs(conscious_effort_force) * energy_cost_multiplier
        self.energy -= energy_cost
        self.energy = max(0.0, self.energy)
        
        # ENV THREAT
        threat_gain = 0.1
        subconscious_reaction_force = environmental_threat * threat_gain
        total_force = conscious_effort_force + subconscious_reaction_force
        random_noise = (np.random.randn() * noise_level) #  # gaussian noise ; noise_lvl is how much random noise
        self.current_arousal += total_force + random_noise
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))
        
        viability_band = [manual_target_arousal - natural_flux, manual_target_arousal + natural_flux]
        
        
        
        # UI controller stuff
        if self.is_tuning and feedback_on:
            pass
        elif controller_type == "P Controller":
            display_kp = kp
            display_ki = 0.0
            display_kd = 0.0
        elif controller_type == "PID Controller":
            display_kp = kp
            display_ki = ki
            display_kd = kd

        
        return self.current_arousal, viability_band, self.fatigue, self.is_burnt_out, self.energy, (display_kp, display_ki, display_kd)