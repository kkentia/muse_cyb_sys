import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from streams.base_stream import BaseStream

class MuseStream(BaseStream):

    def __init__(self, data_queue=None):
        params = BrainFlowInputParams()
        self.board_id = BoardIds.MUSE_2_BOARD.value
        self.board = BoardShim(self.board_id, params)
        
        print("Preparing session... Make sure your Muse is paired and connected.")
        self.board.prepare_session()
        print("Starting stream...")
        self.board.start_stream(450000)
        
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        print("Letting buffer fill...")
        time.sleep(3)

    def get_data(self, noise_level=0):
        data = self.board.get_board_data()
        eeg_data = data[self.eeg_channels]
        return eeg_data
    
    #stops stream and release
    def release(self):
        print("Stopping stream and releasing session...")
        if hasattr(self, 'board') and self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()

    def __del__(self):
        self.release()