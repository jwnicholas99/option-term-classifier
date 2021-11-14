import numpy as np
from sklearn.svm import SVC

from .classifier import Classifier
from utils.plotting import plot_OneClassSVM, plot_TwoClassSVM

class TwoClassSVMClassifier(Classifier):
    def __init__(self, feature_extractor, window_sz=1, gamma='scale'):
        '''
        Args:
            feature_extractor: obj that extracts features by calling extract_features()
        '''
        self.feature_extractor = feature_extractor
        self.window_sz = window_sz
        self.gamma = gamma

    def train(self, X, Y):
        '''
        Train classifier using X and Y. 

        Args:
            states (list(np.array)): list of np.array
            X (list(np.array or MonteRAMState)): list of states
            Y (list(bool)): labels for whether state is a positive eg

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        self.X = X
        self.Y = Y
        self.__update_term_classifier()

    def __update_term_classifier(self):
        states_features = self.feature_extractor.extract_features(self.X)
        feature_matrix = self.__construct_feature_matrix(states_features)

        self.term_classifier = SVC(kernel='rbf', gamma=self.gamma, class_weight='balanced')
        self.term_classifier.fit(feature_matrix, self.Y)

        #plot_TwoClassSVM(self.term_classifier, feature_matrix, f"plots/training_data_windowsz={self.window_sz}_gamma={self.gamma}.png")

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
