import torch
import random
import itertools
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque
from .FullCNN import ImageCNN


class TrainingExample:
    def __init__(self, obs, info):

        self.obs = obs
        self.info = self._construct_info(info)

    def _construct_info(self, x):
        return x if isinstance(x, dict) else dict(player_x=x[0], player_y=x[1])

    @property
    def pos(self):
        pos = self.info['player_x'], self.info['player_y']
        return np.array(pos)

    def __iter__(self):
        """ Allows us to iterate over an object of this class. """
        return ((self.obs, self.info) for _ in [0])


class ConvClassifier:
    """" Generic binary convolutional classifier. """
    def __init__(self,
                device,
                n_input_channels=1,
                batch_size=32):
        
        self.device = device
        self.is_trained = False
        self.batch_size = batch_size

        self.model = ImageCNN(device, n_input_channels, n_classes=1)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Debug variables
        self.losses = []

    @torch.no_grad()
    def predict(self, X):
        if isinstance(X, list):
            X = torch.as_tensor(
                np.array(X)
            ).float().to(self.device)
        return self.model.predict(X)

    def determine_pos_weight(self, y):
        n_negatives = len(y[y != 1])
        n_positives = len(y[y == 1])
        pos_weight = (1. * n_negatives) / n_positives
        return torch.as_tensor(pos_weight).float()

    def should_train(self, y):
        enough_data = len(y) > self.batch_size
        has_positives = len(y[y == 1]) > 0
        has_negatives = len(y[y != 1]) > 0
        return enough_data and has_positives and has_negatives

    def sample(self, X, y):
        idx = random.sample(range(len(X)), k=self.batch_size)
        input_samples = X[idx, :]
        label_samples = y[idx]
        return torch.as_tensor(input_samples).to(self.device),\
               torch.as_tensor(label_samples).to(self.device)

    def fit(self, X, y):
        if self.should_train(y):
            losses = []
            pos_weight = self.determine_pos_weight(y)
            n_gradient_steps = len(X) // self.batch_size

            for _ in tqdm(range(n_gradient_steps)):
                sampled_inputs, sampled_labels = self.sample(X, y)

                logits = self.model(sampled_inputs)
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(),
                                                          sampled_labels,
                                                          pos_weight=pos_weight)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            
            self.is_trained = True

            mean_loss = np.mean(losses)
            self.losses.append(mean_loss)


class EnsembleClassifier(object):
    def __init__(self, ensemble_size, device):
        """ An ensemble of binary convolutional classifiers. """ 

        self.device = device
        self.is_trained = False
        self.ensemble_size = ensemble_size
        
        self.members = [ConvClassifier(device) for _ in range(ensemble_size)]
        self.positive_examples = [deque([], maxlen=100) for _ in range(ensemble_size)]
        self.negative_examples = [deque([], maxlen=100) for _ in range(ensemble_size)]

    @torch.no_grad()
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        predicted_classes = np.vstack(
            [member.predict(X) for member in self.members]
        )
        # managing the case where we have even number of ensemble members
        class_predictions = np.median(predicted_classes, axis=1)
        return (class_predictions > 0.5)

    @torch.no_grad()
    def predict_variance(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        predicted_classes = np.vstack(
            [member.predict(X) for member in self.members]
        )
        return np.var(predicted_classes, axis=1)

    def should_train(self):
        assert len(self.members) > 0
        for i, member in enumerate(self.members):
            Y = self.prepare_training_data(i)[1]
            if not member.should_train(Y):
                return False
        return True

    def prepare_training_data(self, member_idx):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples[member_idx])
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples[member_idx])
        positive_labels = torch.ones((positive_feature_matrix.shape[0],), device=self.device)
        negative_labels = torch.zeros((negative_feature_matrix.shape[0],), device=self.device)

        X = torch.cat((positive_feature_matrix, negative_feature_matrix))
        Y = torch.cat((positive_labels, negative_labels))

        return X, Y

    def construct_feature_matrix(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
        observations = np.array([example.obs for example in examples])
        obs_tensor = torch.as_tensor(observations).float().to(self.device)
        return obs_tensor

    def train(self, X, y):
        self.add_training_data(X, y)
        if self.should_train():
            for i in range(self.ensemble_size):
                X_member, Y_member = self.prepare_training_data(i)
                self.members[i].fit(
                    X_member, Y_member
                )
            
            self.is_trained = True

    def add_training_data(self, X, y):
        for state, label in zip(X, y):
            assert label in (0, 1), label
            f = self.add_positive_trajectory if label else self.add_negative_trajectory
            f([TrainingExample(state, {})])

    def _subsample_trajectory(self, egs):
        subsampled_trajectory = []

        for eg in egs:
            assert isinstance(eg, TrainingExample), type(eg)
            if random.random() > 0.5:
                subsampled_trajectory.append(eg)
        
        return subsampled_trajectory

    def add_positive_trajectory(self, positive_egs):
        for i in range(self.ensemble_size):
            subsampled_trajectory = self._subsample_trajectory(positive_egs)
            self.positive_examples[i].append(subsampled_trajectory)
    
    def add_negative_trajectory(self, negative_egs):
        for i in range(self.ensemble_size):
            subsampled_trajectory = self._subsample_trajectory(negative_egs)
            self.negative_examples[i].append(subsampled_trajectory)
