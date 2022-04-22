import itertools
from utils.monte_preprocessing import parse_ram
from label_extractors.label_extractor import LabelExtractor


class EnsembleOracleExtractor(LabelExtractor):
    '''
    OracleExtractor is given MonteRAMStates and chooses positive egs using labeling_func,
    which might, for example, give a square epsilon around the chosen state
    '''
    def __init__(self, labeling_func):
        '''
        Args:
            extract_only_positive (bool): if true, only return positive egs
        '''
        self.labeling_func = labeling_func

    def extract_labels(self, state_trajs, raw_ram_trajs, subgoal_info):
        '''
        Extract labels from a given state trajectory, raw ram state trajectory and the idx of the subgoal.
        As of now, OracleExtractor only works with single frames as states rather than frame stacks.

        Note that the OracleExtractor has 2 classes of labels:
            1. Positive, in subgoal trajectory (1)
            2. Negative, in subgoal trajectory (0)

        Args:
            state_trajs (list (list(np.array))): state trajectories
            raw_ram_trajs (list (list(np.array))): state trajectories - RawRAM states
            subgoal_info (tuple): x, y, room_number

        Returns:
            (list(np.array)): list of np.array of states
            (list(int)): list of labels of corresponding states
        '''
        def flatten(x):
            return list(itertools.chain.from_iterable(x))

        pos_states, neg_states = [], []
        states = flatten(state_trajs)
        rams = flatten(raw_ram_trajs)

        for state, ram in zip(states, rams):
            if self.labeling_func(subgoal_info, parse_ram(ram)):
                pos_states.append(state)
            else:
                continue  # TODO: HACK
                neg_states.append(state)

        states = pos_states + neg_states
        labels = [1 for _ in range(len(pos_states))] + [0 for _ in range(len(neg_states))]

        return states, labels
