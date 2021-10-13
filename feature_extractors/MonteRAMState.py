from utils.monte_preprocessing import parse_ram
from .feature_extractor import FeatureExtractor

class MonteRAMState(FeatureExtractor):
    def extract_features(self, states):
        '''
        Extract MonteRAMState from raw RAM

        Args:
            states (list(np.array)): list of np.array

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        return states
