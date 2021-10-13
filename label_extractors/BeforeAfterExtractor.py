from label_extractors.label_extractor import LabelExtractor

class BeforeAfterExtractor(LabelExtractor):
    def __init__(self, extract_only_pos=False, window_sz=5):
        '''
        Args:
            extract_only_positive (bool): if true, only return positive egs
            window_sz (int): how many states before or after to count as positive egs
        '''
        self.extract_only_pos = extract_only_pos
        self.window_sz = window_sz

    def extract_labels(self, state_traj, idx):
        '''
        Extract labels from a given state trajectory and the idx of the subgoal

        Args:
            state_traj (list(np.array)): state trajectory - can be RAM, frames or other reprs
            idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of positive states
            (list(int)): list of indices of positive states
            (list(np.array)): list of np.array of negative states
            (list(int)): list of indices of negative states
        '''
        pos_start = max(0, idx - self.window_sz)
        pos_end = min(len(state_traj), idx + self.window_sz)

        pos_idxs = list(range(pos_start, pos_end + 1))
        pos_states = [state_traj[i] for i in pos_idxs]

        if not self.extract_only_pos:
            neg_idxs = [i for i in range(len(state_traj)) if i < pos_start or i > pos_end]
            neg_states = [state_traj[i] for i in neg_idxs]
        else:
            neg_idxs, neg_states = [], []

        return pos_states, pos_idxs, neg_states, neg_idxs
