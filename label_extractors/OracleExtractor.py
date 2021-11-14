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

    def extract_labels(self, state_trajs, subgoal_traj_idx, subgoal_state_idx):
        '''
        Extract labels from a given raw ram state trajectory and the idx of the subgoal

        Args:
            state_traj (list (list(np.array))): state trajectories - must be Raw ram state
            subgoal_traj_idx (int): index of traj containing the subgoal
            subgoal_state_idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of positive states
            (list(np.array)): list of np.array of negative states
        '''
        pos_states, neg_states = [], []

        subgoal_traj = state_trajs[subgoal_traj_idx]
        subgoal = parse_ram(subgoal_traj[subgoal_state_idx])
        for i, state in enumerate(subgoal_traj):
            if self.labeling_func(subgoal, parse_ram(state)):
                pos_states.append(state)
            elif not self.extract_only_pos:
                neg_states.append(state)
        return pos_states, neg_states
