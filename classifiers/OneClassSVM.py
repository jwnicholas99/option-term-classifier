import numpy as np
from sklearn.svm import OneClassSVM

from .classifier import Classifier

class OneClassSVMClassifier(Classifier):
    def __init__(self, feature_extractor):
        '''
        Args:
            feature_extractor: obj that extracts features by calling extract_features()
        '''
        self.feature_extractor = feature_extractor

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

    def __update_term_classifier(self, nu=0.1):
        states_features = self.feature_extractor.extract_features(self.term_set)
        positive_feature_matrix = self.__construct_feature_matrix(states_features)
        self.term_classifier = OneClassSVM(kernel='rbf', nu=nu)
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
