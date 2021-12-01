from label_extractors.label_extractor import LabelExtractor

class AfterExtractor(LabelExtractor):
    def __init__(self, extract_only_pos=False, window_sz=5):
        '''
        Args:
            extract_only_positive (bool): if true, only return positive egs
            window_sz (int): how many states before or after to count as positive egs
        '''
        self.extract_only_pos = extract_only_pos
        self.window_sz = window_sz

    def extract_labels(self, state_trajs, subgoal_traj_idx, subgoal_state_idx):
        '''
        Extract labels from a given state trajectory and the idx of the subgoal.

        Note that the AfterExtractor has 2 classes of labels:
            1. Positive, in subgoal trajectory (1)
            2. Negative, in subgoal trajectory (0)

        Args:
            state_traj (list (list(np.array))): state trajectories
            subgoal_traj_idx (int): index of traj containing the subgoal
            subgoal_state_idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of states
            (list(int)): list of labels of corresponding states
        '''
        subgoal_traj = state_trajs[subgoal_traj_idx]

        pos_start = subgoal_state_idx
        pos_end = min(len(subgoal_traj), subgoal_state_idx + self.window_sz)

        pos_idxs = list(range(pos_start, pos_end + 1))
        pos_states = [subgoal_traj[i] for i in pos_idxs]

        if not self.extract_only_pos:
            neg_idxs = [i for i in range(len(subgoal_traj)) if i < pos_start or i > pos_end]
            neg_states = [subgoal_traj[i] for i in neg_idxs]
        else:
            neg_states = []

        states = pos_states + neg_states
        labels = [1 for _ in range(len(pos_states))] + [0 for _ in range(len(neg_states))]

        return states, labels
