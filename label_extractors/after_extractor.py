from .label_extractor import LabelExtractor


class AfterExtractor(LabelExtractor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, state_trajectory):
        return super().__call__(state_trajectory)

    def extract_after(self, state_trajectory, start_index, num_after):
        return state_trajectory[start_index: start_index + num_after]
