from label_extractors.label_extractor import LabelExtractor

class TransductiveExtractor(LabelExtractor):
    def __init__(self, extract_only_pos=False, window_sz=5, num_trajs=10):
        '''
        Args:
            extract_only_positive (bool): if true, only return positive egs
            window_sz (int): how many states before or after to count as positive egs
        '''
        self.extract_only_pos = extract_only_pos
        self.window_sz = window_sz
        self.num_trajs = num_trajs

    def extract_labels(self, state_trajs, subgoal_traj_idx, subgoal_state_idx):
        '''
        Extract labels from a given state trajectory and the idx of the subgoal.

        Note that the TransductiveExtractor has 3 classes of labels:
            1. Positive, in subgoal trajectory (1)
            2. Negative, in subgoal trajectory (0)
            3. Negative, all states outside of subgoal trajectory (-1)

        Args:
            state_traj (list (list(np.array))): state trajectories
            subgoal_traj_idx (int): index of traj containing the subgoal
            subgoal_state_idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of states
            (list(int)): list of labels of corresponding states
        '''
        subgoal_traj = state_trajs[subgoal_traj_idx]

        pos_start = max(0, subgoal_state_idx - self.window_sz)
        pos_end = min(len(subgoal_traj), subgoal_state_idx + self.window_sz)

        pos_idxs = list(range(pos_start, pos_end + 1))
        pos_states = [subgoal_traj[i] for i in pos_idxs]

        if not self.extract_only_pos:
            subgoal_neg_idxs = [i for i in range(len(subgoal_traj)) if i < pos_start or i > pos_end]
            subgoal_neg_states = [subgoal_traj[i] for i in subgoal_neg_idxs] 

            non_subgoal_trajs_start = max(0, subgoal_traj_idx - self.num_trajs)
            non_subgoal_trajs = [state_trajs[i] for i in range(non_subgoal_trajs_start, subgoal_traj_idx)]
            non_subgoal_neg_states = [state for non_subgoal_traj in non_subgoal_trajs for state in non_subgoal_traj]
        else:
            subgoal_neg_states = []
            non_subgoal_neg_states = []

        states = pos_states + subgoal_neg_states + non_subgoal_neg_states
        labels = [1 for _ in range(len(pos_states))] \
            + [0 for _ in range(len(subgoal_neg_states))] \
            + [-1 for _ in range(len(non_subgoal_neg_states))]

        return states, labels
