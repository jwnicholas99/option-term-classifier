import numpy as np
import cv2

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
            downsampled = cv2.resize(state, (52, 52), interpolation=cv2.INTER_LINEAR)
            output.append(downsampled)

        return output
