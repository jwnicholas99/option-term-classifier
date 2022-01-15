import gzip
import pickle
from tqdm import tqdm

from label_extractors.labeling_funcs import square_epsilon, square_epsilon_screen

from utils.monte_preprocessing import parse_ram

class Data():
    def __init__(self, path, train_skip, train_num, test_skip, test_num):
        self.data_path = path
        self.traj_generator = self.load_trajs(skip=0)

        self.train_raw_ram_trajs, self.train_ram_trajs, self.train_frame_trajs = self.parse_trajs(skip=train_skip, num_trajs=train_num)
        self.test_raw_ram_trajs, self.test_ram_trajs, self.test_frame_trajs = self.parse_trajs(skip=test_skip, num_trajs=test_num)

    def load_trajs(self, skip=0):
        '''
        Returns a generator for getting states.

        Args:
            skip (int): number of trajectories to skip

        Returns:
            (generator): generator to be called for trajectories
        '''
        print(f"[+] Loading trajectories from file '{self.data_path}'")
        with gzip.open(self.data_path, 'rb') as f:
            print(f"[+] Skipping {skip} trajectories...")
            for _ in tqdm(range(skip)):
                traj = pickle.load(f)

            try:
                while True:
                    traj = pickle.load(f)
                    yield traj
            except EOFError:
                pass

    def parse_trajs(self, skip, num_trajs):
        '''
        Parse and separate trajectories into ram trajectories and frame trajectories.
        Note that because trajs is a generator, sequential calls to parse_trajs() will have
        sequential trajs

        Args:
            start (int): starting traj index to parse (inclusive)
            end (int): end traj index to parse (inclusive)

        Returns:
            (List of lists of raw RAM, List of lists of MonteRAMState, List of lists of frames)
        '''
        for _ in range(skip):
            next(self.trajs_generator)

        raw_ram_trajs, ram_trajs, frame_trajs = [], [], []
        for i in range(num_trajs):
            traj = next(self.traj_generator)

            raw_ram_traj, ram_traj, frame_traj = [], [], []
            for ram, frame in traj:
                raw_ram_traj.append(ram)
                ram_traj.append(parse_ram(ram))
                frame_traj.append(frame)
            raw_ram_trajs.append(raw_ram_traj)
            ram_trajs.append(ram_traj)
            frame_trajs.append(frame_traj)
        return raw_ram_trajs, ram_trajs, frame_trajs

    def find_first_instance(self, trajs, subgoal):
        for i, traj in enumerate(trajs):
            for j, state in enumerate (traj):
                if state.player_x == subgoal[0] and state.player_y == subgoal[1] and state.screen == subgoal[2]:
                    return i, j
        return None, None

    def filter_in_term_set(self, trajs, subgoal):
        '''
        Filters given ram states to return a set of indices of the states that we consider to be in the 
        termination set of the given subgoal

        Args:
            trajs (list of lists of MonteRAMState): a list of lists of ram states to be filtered
            subgoal (MonteRAMState): the ram state chosen to be subgoal

        Returns:
            (list of (int, int)): list of indices in the termination set of the subgoal
        '''
        term_set = []
        for i, traj in enumerate(trajs):
            for j, state in enumerate(traj):
                if square_epsilon_screen(subgoal, state):
                    term_set.append((i, j))
        return term_set
