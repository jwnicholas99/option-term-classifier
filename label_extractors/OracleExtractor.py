from utils.monte_preprocessing import parse_ram
from label_extractors.label_extractor import LabelExtractor

class OracleExtractor(LabelExtractor):
    '''
    OracleExtractor is given MonteRAMStates and chooses positive egs using labeling_func,
    which might, for example, give a square epsilon around the chosen state
    '''
    def __init__(self, labeling_func, extract_only_pos=False):
        '''
        Args:
            extract_only_positive (bool): if true, only return positive egs
            labeling_func ((subgoal, state) -> bool): function for labeling states as positive or negative egs
        '''
        self.extract_only_pos = extract_only_pos
        self.labeling_func = labeling_func

    def extract_labels(self, state_trajs, raw_ram_trajs, subgoal_traj_idx, subgoal_state_idx):
        '''
        Extract labels from a given state trajectory, raw ram state trajectory and the idx of the subgoal.
        As of now, OracleExtractor only works with single frames as states rather than frame stacks.

        Note that the OracleExtractor has 2 classes of labels:
            1. Positive, in subgoal trajectory (1)
            2. Negative, in subgoal trajectory (0)

        Args:
            state_trajs (list (list(np.array))): state trajectories
            raw_ram_trajs (list (list(np.array))): state trajectories - RawRAM states
            subgoal_traj_idx (int): index of traj containing the subgoal
            subgoal_state_idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of states
            (list(int)): list of labels of corresponding states
        '''
        pos_states, neg_states = [], []

        raw_ram_subgoal_traj = raw_ram_trajs[subgoal_traj_idx]
        state_subgoal_traj = state_trajs[subgoal_traj_idx]

        subgoal = parse_ram(raw_ram_subgoal_traj[subgoal_state_idx])
        for i, raw_ram_state in enumerate(raw_ram_subgoal_traj):
            if self.labeling_func(subgoal, parse_ram(raw_ram_state)):
                pos_states.append(state_subgoal_traj[i])
            elif not self.extract_only_pos:
                neg_states.append(state_subgoal_traj[i])

        states = pos_states + neg_states
        labels = [1 for _ in range(len(pos_states))] + [0 for _ in range(len(neg_states))]

        return states, labels
