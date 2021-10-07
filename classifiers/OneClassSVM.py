import numpy as np
from sklearn.svm import OneClassSVM

class OneClassSVMClassifier():
    def __init__(self, term_set, feature_extractor):
        self.term_set = term_set
        self.feature_extractor = feature_extractor
        self.update_term_classifier()

    def update_term_classifier(self, nu=0.1):
        states_features = self.feature_extractor.extract_features(self.term_set)
        positive_feature_matrix = self.construct_feature_matrix(states_features)
        self.term_classifier = OneClassSVM(kernel='rbf', nu=nu)
        self.term_classifier.fit(positive_feature_matrix)

    def construct_feature_matrix(self, states_features):
        return (np.array([np.reshape(state_features, (-1,)) for state_features in states_features]))

    def is_term(self, state):
        features = np.reshape(np.array(self.feature_extractor.extract_features([state])), (1, -1))
        return self.term_classifier.predict(features)[0] == 1
