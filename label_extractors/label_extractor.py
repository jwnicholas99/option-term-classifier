class LabelExtractor:
    def __init__(self, extract_only_positive) -> None:
        pass

    def __call__(self, state_trajectory):
        positive_states = []
        positive_indices = []

        negative_states = []
        negative_indices = []

        return positive_states, positive_indices, \
               negative_states, negative_indices
