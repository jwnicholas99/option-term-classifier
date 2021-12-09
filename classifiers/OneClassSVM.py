import numpy as np
from sklearn.svm import OneClassSVM

from .classifier import Classifier
from utils.plotting import plot_SVM

class OneClassSVMClassifier(Classifier):
    def __init__(self, feature_extractor, window_sz=1, nu=0.1, gamma='scale'):
        '''
        Args:
            feature_extractor: obj that extracts features by calling extract_features()
        '''
        self.feature_extractor = feature_extractor
        self.window_sz = window_sz
        self.nu = nu
        self.gamma = gamma

    def train(self, X, Y):
        '''
        Train classifier using X and Y. Negative examples are ignored as this is a
        one class SVM

        Args:
            states (list(np.array)): list of np.array
            X (list(np.array or MonteRAMState)): list of states
            Y (list(bool)): labels for whether state is a positive eg

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        self.term_set = [x for i, x in enumerate(X) if Y[i]]
        self.__update_term_classifier()

    def __update_term_classifier(self):
        states_features = self.feature_extractor.extract_features(self.term_set)
        positive_feature_matrix = self.__construct_feature_matrix(states_features)
        self.term_classifier = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
        self.term_classifier.fit(positive_feature_matrix)

    def __construct_feature_matrix(self, states_features):
        return (np.array([np.reshape(state_features, (-1,)) for state_features in states_features]))

    def predict(self, state):
        '''
        Predict whether state is in the term set

        Args:
            state (np.array or MonteRAMState): chosen state

        Returns:
            (bool): whether state is in the term set
        '''
        features = np.reshape(np.array(self.feature_extractor.extract_features([state])), (1, -1))
        return self.term_classifier.predict(features)[0] == 1

    def predict_raw(self, state):
        '''
        Predict class label of state (either 0 or 1 as OneClassSVM only has 2 classes)

        Args:
            state (np.array or MonteRAMState): chosen state

        Returns:
            (int): predicted class label of state
        '''
        features = np.reshape(np.array(self.feature_extractor.extract_features([state])), (1, -1))
        return self.term_classifier.predict(features)[0]
