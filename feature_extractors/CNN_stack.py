import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchsummary import summary

from utils.monte_preprocessing import parse_ram
from .feature_extractor import FeatureExtractor

class CNN_stack(FeatureExtractor):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        '''
        # 1 Conv Layer
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=10,
                stride=5),
        )
        '''

        '''
        # 2 Conv Layers
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
        )
        '''
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.MaxPool2d(
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(3136, 512)
        )

        self.model.cuda()
        summary(self.model, (1, 84, 84))

        for p in self.model:
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.model.parameters():
            param.requires_grad = False

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

            batch_features = self.model(torch.from_numpy(flattened_images).float().to("cuda:0"))
            batch_features = batch_features.cpu().numpy()
            batch_features = np.split(batch_features, indices_or_sections=batch_sz)

            output.extend(batch_features)

            #batch_states = batch_states.transpose(0, 3, 1, 2)
        return output

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
