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

    def extract_labels(self, state_traj, subgoal):
        '''
        Extract labels from a given raw ram state trajectory and the idx of the subgoal

        Args:
            state_traj (list(np.array)): state trajectory - must be Raw ram state
            idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of positive states
            (list(int)): list of indices of positive states
            (list(np.array)): list of np.array of negative states
            (list(int)): list of indices of negative states
        '''
        pos_states, pos_idxs, neg_states, neg_idxs = [], [], [], []
        subgoal = parse_ram(subgoal)

        for i, state in enumerate(state_traj):
            if self.labeling_func(subgoal, parse_ram(state)):
                pos_states.append(state)
                pos_idxs.append(i)
            elif not self.extract_only_pos:
                neg_states.append(state)
                neg_idxs.append(i)
        return pos_states, pos_idxs, neg_states, neg_idxs
