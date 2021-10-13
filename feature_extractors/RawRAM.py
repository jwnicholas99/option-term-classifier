from .feature_extractor import FeatureExtractor

class RawRAM(FeatureExtractor):
    def extract_features(self, states):
        '''
        Returns RAM state without extracting feautures

        Args:
            states (list(np.array)): list of np.array

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        return states
