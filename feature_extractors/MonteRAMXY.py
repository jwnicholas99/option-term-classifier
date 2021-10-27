import numpy as np

from utils.monte_preprocessing import parse_ram_xy
from .feature_extractor import FeatureExtractor

class MonteRAMXY(FeatureExtractor):
    def extract_features(self, states):
        '''
        Extract x and y position from raw RAM

        Args:
            states (list(np.array)): list of np.array

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        return [np.asarray(parse_ram_xy(state)) for state in states]
