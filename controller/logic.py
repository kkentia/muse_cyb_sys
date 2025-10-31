#FEEDBACK LOGIC regulator

class Controller:
    def __init__(self):
        self.in_range = False
        self.last_good_arousal = 0.5
        self.hysteresis_counter = 0
        self.hysteresis_threshold = 30  # needs 30 consecutive out-of-range samples to change state -> 3 sec

    def update_state(self, arousal_index, viability_band, artifact_detected):
        #updates sys based on current arousal
        if artifact_detected:
            return self.in_range, self.last_good_arousal  #last good state

        if arousal_index is None:
            return self.in_range, self.last_good_arousal

        self.last_good_arousal = arousal_index
        lower_bound, upper_bound = viability_band

        if lower_bound <= arousal_index <= upper_bound:
            self.hysteresis_counter = 0
            self.in_range = True
        else:
            self.hysteresis_counter += 1

        if self.hysteresis_counter >= self.hysteresis_threshold:
            self.in_range = False

        return self.in_range, self.last_good_arousal