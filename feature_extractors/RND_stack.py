import numpy as np
import torch

from rnd.model import RNDModel

from utils.monte_preprocessing import parse_ram
from .feature_extractor import FeatureExtractor

class RND_stack(FeatureExtractor):
    def __init__(self, predictor_path, batch_size=32):
        self.batch_size = batch_size
        self.rnd = RNDModel((4, 84, 84), 18)
        self.rnd.predictor.load_state_dict(torch.load(predictor_path))
        self.rnd.predictor.cuda()

        def get_features(model, input, output):
            self.features = output.detach()

        for param in self.rnd.predictor.parameters():
            param.requires_grad = False
        #self.rnd.predictor[4].register_forward_hook(get_features)

    def extract_features(self, states):
        '''
        Extract RND features from raw images

        Args:
            states (list(np.array)): list of np.array

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        output = []
        for i in range(0, len(states), self.batch_size):
            batch_images = states[i:i+self.batch_size]
            batch_sz = len(batch_images)
            flattened_images = [frame for frame_stack in batch_images for frame in frame_stack]
            flattened_images = np.stack(flattened_images, axis=0)
            flattened_images = flattened_images.transpose(0, 3, 1, 2)

            batch_features = self.rnd.predictor(torch.from_numpy(flattened_images).float().to("cuda:0"))
            batch_features = batch_features.cpu().numpy()
            batch_features = np.split(batch_features, indices_or_sections=batch_sz)

            output.extend(batch_features)
        return output
