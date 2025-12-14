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
        self.burnout_recovery_target = 0  # go to 0 arousal
        self.energy = 1.0

        # PID state vars
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.kp = 0.1; self.ki = 0.0; self.kd = 0.0


        self.is_tuning = False
        self.tuning_kp = 0.0
        self.tuning_peak_times = []
        self.tuning_peak_values = []
        
        #adaptive vars for PID
        self.out_of_band_counter = 0
        self.adaptive_multiplier = 1.0


    def reset(self, state_name):
        self.current_arousal = self.states[state_name]['initial_arousal']
        self.arousal_history.clear()
        self.fatigue = 0.0; self.outer_loop_counter = 0; self.target_override = 0.0
        self.is_burnt_out = False; self.energy = 1.0
        self.integral_error = 0.0; self.previous_error = 0.0
        self.kp = 0.1; self.ki = 0.0; self.kd = 0.0
        self.is_tuning = False; self.tuning_kp = 0.0
        self.tuning_peak_times = []; self.tuning_peak_values = []
        self.out_of_band_counter = 0; self.adaptive_multiplier = 1.0
        self.burnout_recovery_target = 0.0

    #perturbation
    def apply_spike(self, spike_magnitude):
        self.current_arousal += spike_magnitude
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))

    def start_auto_tuning(self):
        print("--- STARTING AUTO-TUNER ---")
        self.is_tuning = True
        self.tuning_kp = 0.1  # uultimate gain
        self.integral_error = 0.0; self.previous_error = 0.0
        self.tuning_peak_times = []; self.tuning_peak_values = []



    # MAIN METHOD------------------------
    def get_arousal_value(self, state_name, manual_target_arousal, natural_flux, noise_level, kp, ki, kd, control_delay, feedback_on, environmental_threat, effort_amplification, controller_type):
        
        # feedback off = no conscious control
        if not feedback_on:
            # energy regen
            energy_regeneration_rate = 0.001
            if self.current_arousal < 0.4: energy_regeneration_rate *= 2
            self.energy += energy_regeneration_rate
            self.energy = min(1.0, self.energy)
            
            # fatigue recover 

            self.outer_loop_counter += 1
            if self.outer_loop_counter >= 20:
                self.outer_loop_counter = 0
                self.fatigue -= 0.1
                self.fatigue = max(0.0, self.fatigue)
            
            self.arousal_history.append(self.current_arousal)
            
            #reset ctrl states
            self.integral_error = 0.0
            self.previous_error = 0.0
            self.out_of_band_counter = 0
            self.is_burnt_out = False
            
            if self.is_tuning:
                print("Feedback turned OFF - stopping auto-tuner")
                self.is_tuning = False
            
            # natural drift to resting state (weak)
            resting_target = self.states[state_name]['initial_arousal']
            drift_error = resting_target - self.current_arousal
            conscious_effort_force = drift_error * 0.02
            
            #only external forces left
            threat_gain = 0.1
            subconscious_reaction_force = environmental_threat * threat_gain
            total_force = conscious_effort_force + subconscious_reaction_force
            random_noise = (np.random.randn() * noise_level)
            
            self.current_arousal += total_force + random_noise
            self.current_arousal = max(0.0, min(1.0, self.current_arousal))
            
            viability_band = [manual_target_arousal - natural_flux, manual_target_arousal + natural_flux]
            
            #display
            return self.current_arousal, viability_band, self.fatigue, self.is_burnt_out, self.energy, (0.0, 0.0, 0.0)
        
        
        
        
#------------------------   feedback_on TRUE:
        
        # 1) energy regen
        energy_regeneration_rate = 0.001
        if self.current_arousal < 0.4: energy_regeneration_rate *= 2
        self.energy += energy_regeneration_rate
        self.energy = min(1.0, self.energy)

        # 2) fatigue & burnout logic
        self.outer_loop_counter += 1
        if self.outer_loop_counter >= 20:
            self.outer_loop_counter = 0
            if not self.is_burnt_out:
                if self.current_arousal > 0.6: self.fatigue += 0.05
                elif self.current_arousal < 0.4: self.fatigue -= 0.1
            else:
                if self.current_arousal < 0.4:
                    self.fatigue -= 0.01
            self.fatigue = max(0.0, min(1.0, self.fatigue))
            if self.fatigue >= 1.0: self.is_burnt_out = True
            if self.is_burnt_out and self.fatigue <= 0.0: self.is_burnt_out = False
            if self.fatigue > 0.7 and not self.is_burnt_out:
                self.target_override = (0.25 - manual_target_arousal) * 0.1 
            else: self.target_override = 0.0
        
        self.arousal_history.append(self.current_arousal)
        
        # calculate error (w & w/out delay )
        effective_target = manual_target_arousal + self.target_override
        target = effective_target
        error = target - self.current_arousal

        # adaptive control
        if not self.is_tuning and not self.is_burnt_out:
            viability_band = [manual_target_arousal - natural_flux, manual_target_arousal + natural_flux]
            if self.current_arousal < viability_band[0] or self.current_arousal > viability_band[1]:
                self.out_of_band_counter += 1
            else:
                self.out_of_band_counter = 0
                if self.adaptive_multiplier > 1.0: self.adaptive_multiplier -= 0.005

            if self.out_of_band_counter > 40: # >2 seconds out of target VB
                self.adaptive_multiplier += 0.02
            self.adaptive_multiplier = min(5.0, max(1.0, self.adaptive_multiplier))




        # 3) main force ctrl
        conscious_effort_force = 0.0
        display_kp = 0.0; display_ki = 0.0; display_kd = 0.0

        # burnout recovery
        if self.is_burnt_out:
            conscious_effort_force = (self.burnout_recovery_target - self.current_arousal) * 0.05
            # Reset internal PID states during burnout
            self.integral_error = 0.0
            self.previous_error = 0.0

        # autotune
        elif self.is_tuning:
            tuning_error = manual_target_arousal - self.current_arousal
            self.tuning_kp += 0.005 
            conscious_effort_force = tuning_error * self.tuning_kp
        
            display_kp = self.tuning_kp

            #detect peaks in arousal to determine ultimate gain and period
            history = np.array(self.arousal_history)
            if len(history) > 5:
                smooth_curr = np.mean(history[-3:])
                smooth_prev = np.mean(history[-4:-1])
                smooth_prev2 = np.mean(history[-5:-2])

                is_peak = (smooth_prev > smooth_prev2) and (smooth_prev > smooth_curr)
                
                if is_peak and smooth_prev > manual_target_arousal:
                    self.tuning_peak_times.append(len(history))
                    self.tuning_peak_values.append(smooth_prev)
                    
                    if len(self.tuning_peak_times) > 3:
                        avg_amplitude = np.mean(self.tuning_peak_values[-3:])
                        if abs(avg_amplitude - manual_target_arousal) > 0.02:
                            dt = 0.05 
                            tu_samples = np.mean(np.diff(self.tuning_peak_times[-3:]))
                            tu_seconds = tu_samples * dt
                            
                        # Ziegler-Nichols PID equatioms
                            ku = self.tuning_kp
                            self.kp = 0.6 * ku
                            self.ki = (2.0 * self.kp) / tu_seconds
                            self.kd = (self.kp * tu_seconds) / 8.0
                            
                            print(f"--- TUNING COMPLETE --- Ku={ku:.3f}, Tu={tu_seconds:.2f}s")
                            print(f"New Gains: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}")
                            self.is_tuning = False

        # 4) choose controller
        else:
            pid_force = 0.0
            
            # safety to 0 values
            base_kp = kp if kp > 0 else self.kp
            base_ki = ki if ki > 0 else self.ki
            base_kd = kd if kd > 0 else self.kd

            if controller_type == "P Controller":
                active_kp = base_kp * self.adaptive_multiplier
                pid_force = active_kp * error
                
                display_kp = active_kp
                self.integral_error = 0.0
                self.previous_error = error

            elif controller_type == "PID Controller":
                active_kp = base_kp * self.adaptive_multiplier
                active_ki = base_ki * self.adaptive_multiplier
                active_kd = base_kd * self.adaptive_multiplier

                self.integral_error += error
                self.integral_error = max(-5.0, min(5.0, self.integral_error))
                derivative_error = error - self.previous_error
                self.previous_error = error

                pid_force = (active_kp * error) + (active_ki * self.integral_error) + (active_kd * derivative_error)
                
                display_kp = active_kp
                display_ki = active_ki
                display_kd = active_kd

            # apply amp, fatigue, and energy  --> limits
            amplifier = 1.0 + (abs(error) * effort_amplification)
            amplified_force = pid_force * amplifier
            
            energy_limitation_factor = self.energy
            fatigue_degradation_factor = 1.0 - ((self.fatigue - 0.7) / 0.3) if self.fatigue > 0.7 else 1.0
            
            conscious_effort_force = amplified_force * fatigue_degradation_factor * energy_limitation_factor

        # 5) energy cost and external forces
        energy_cost_multiplier = 0.1
        energy_cost = abs(conscious_effort_force) * energy_cost_multiplier
        self.energy -= energy_cost
        self.energy = max(0.0, self.energy)

        threat_gain = 0.1
        subconscious_reaction_force = environmental_threat * threat_gain
        total_force = conscious_effort_force + subconscious_reaction_force
        random_noise = (np.random.randn() * noise_level)
        
        self.current_arousal += total_force + random_noise
        self.current_arousal = max(0.0, min(1.0, self.current_arousal))

        viability_band = [manual_target_arousal - natural_flux, manual_target_arousal + natural_flux]

        return self.current_arousal, viability_band, self.fatigue, self.is_burnt_out, self.energy, (display_kp, display_ki, display_kd)