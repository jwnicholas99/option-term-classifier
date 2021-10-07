from .label_extractor import LabelExtractor


class OracleExtractor(LabelExtractor):
    def __init__(self, extract_only_positive, labeling_function) -> None:
        self.labeling_function = labeling_function
        super().__init__(extract_only_positive)

    def __call__(self, state_trajectory):
        for state in state_trajectory:
            if self.labeling_function(state):
                positives.add(state)