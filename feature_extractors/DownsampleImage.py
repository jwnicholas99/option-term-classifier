import numpy as np

from skimage.measure import block_reduce

class DownsampleImage():
    def extract_features(self, states):
        '''
        Returns image state after downsampling

        Args:
            states (list(np.array)): list of np.array

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        output = []
        for state in states:
            downsampled = block_reduce(state, block_size=(1, 2, 2), func=np.mean)
            output.append(downsampled)

        return output
