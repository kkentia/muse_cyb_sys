from abc import ABC, abstractmethod

class BaseStream(ABC):
    @abstractmethod
    def get_data(self, noise_level=0.1): 
        pass