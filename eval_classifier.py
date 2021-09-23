import argparse
import gzip
import pickle
import numpy as np
from tqdm import tqdm

from classifiers.OneClassSVM import OneClassSVMClassifier
from feature_extractors.RawImage import RawImage
from utils.monte_preprocessing import parse_ram

def load_trajectories(path, skip=0):
    '''
    Returns a generator for getting states.

    Args:
        path (str): filepath of pkl file containing trajectories
        skip (int): number of trajectories to skip

    Returns:
        (generator): generator to be called for trajectories
    '''
    print(f"[+] Loading trajectories from file '{path}'")
    with gzip.open(path, 'rb') as f:
        for _ in range(skip):
            traj = pickle.load(f)

        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass

def parse_trajectories(trajs):
    '''
    Parse and separate trajectories into ram trajectories and frame trajectories

    Args:
        trajs (generator): trajectories containing ram and frame states

    Returns:
        (List of lists of MonteRAMState, List of lists of frames)
    '''
    ram_trajs, frame_trajs = [], []
    for traj in trajs:
        ram_traj, frame_traj = [], []
        for ram, frame in traj:
            ram_traj.append(parse_ram(ram))
            frame_traj.append(frame)
        ram_trajs.append(ram_traj)
        frame_trajs.append(frame_traj)
    return ram_trajs, frame_trajs

def filter_in_term_set(trajs, subgoal):
    '''
    Filters given ram states to return a set of indices of the states that we consider to be in the 
    termination set of the given subgoal

    Args:
        trajs (list of lists of MonteRAMState): a list of lists of ram states to be filtered
        subgoal (MonteRAMState): the ram state chosen to be subgoal

    Returns:
        (set of (int, int)): set of indices in the termination set of the subgoal
    '''
    term_set = set()
    for traj_idx in range(len(trajs)):
        for state_idx in range(len(trajs[traj_idx])):
            state = trajs[traj_idx][state_idx]
            if (
                abs(state.player_x - subgoal.player_x) <= 1 and
                abs(state.player_y - subgoal.player_y) <= 1 and
                state.has_key == subgoal.has_key
            ):
                term_set.add((traj_idx, state_idx))
    return term_set

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate termination classifier performance')

    parser.add_argument('filepath', type=str, help='filepath of pkl file containing trajectories with RAM states and frames')
    parser.add_argument('term_classifier', type=str, choices=['OneClassSVM'], help='termination classifier to be used')
    parser.add_argument('feature_extractor', type=str, choices=['RawImage'], help='feature extractor to be used')

    args = parser.parse_args()

    trajs_generator = load_trajectories(args.filepath)
    ram_trajs, frame_trajs = parse_trajectories(trajs_generator)

    traj_idx = 1300
    state_idx = 213
    window_sz = 5

    subgoal_ram = ram_trajs[traj_idx][state_idx]
    term_set = filter_in_term_set(ram_trajs, subgoal_ram)

    if args.feature_extractor == 'RawImage':
        feature_extractor = RawImage()

    frame_term_set = frame_trajs[traj_idx][state_idx - window_sz: state_idx + window_sz]
    if args.term_classifier == 'OneClassSVM':
        term_classifier = OneClassSVMClassifier(frame_term_set, feature_extractor)

    output = set()
    for traj_idx in range(len(frame_trajs)):
        for state_idx in range(len(frame_trajs[traj_idx])):
            if term_classifier.is_term(np.array(frame_trajs[traj_idx][state_idx])):
                output.add((traj_idx, state_idx))

    print(len(term_set.intersection(output)))
