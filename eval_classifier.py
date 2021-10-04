import argparse
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classifiers.OneClassSVM import OneClassSVMClassifier
from feature_extractors.RawImage import RawImage
from feature_extractors.DownsampleImage import DownsampleImage
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
        print(f"[+] Skipping {skip} trajectories...")
        for _ in tqdm(range(skip)):
            traj = pickle.load(f)

        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass

def parse_trajectories(trajs, start, end):
    '''
    Parse and separate trajectories into ram trajectories and frame trajectories

    Args:
        trajs (generator): trajectories containing ram and frame states

    Returns:
        (List of lists of MonteRAMState, List of lists of frames)
    '''
    ram_trajs, frame_trajs = [], []
    for i, traj in enumerate(trajs):
        if i < start:
            continue
        if i > end:
            break

        ram_traj, frame_traj = [], []
        for ram, frame in traj:
            ram_traj.append(parse_ram(ram))
            frame_traj.append(frame)
        ram_trajs.append(ram_traj)
        frame_trajs.append(frame_traj)
    return ram_trajs, frame_trajs

def find_first_instance(trajs, subgoal):
    for i, traj in enumerate(trajs):
        for j, state in enumerate (traj):
            if state.player_x == subgoal[0] and state.player_y == subgoal[1]:
                return i, j

def filter_in_term_set(trajs, subgoal):
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
        for j, state in enumerate(trajs[traj_idx]):
            if (
                abs(state.player_x - subgoal.player_x) <= 1 and
                abs(state.player_y - subgoal.player_y) <= 1 and
                state.has_key == subgoal.has_key
            ):
                term_set.append((i, j))
    return term_set

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate termination classifier performance')

    parser.add_argument('filepath', type=str, help='filepath of pkl file containing trajectories with RAM states and frames')
    parser.add_argument('term_classifier', type=str, choices=['OneClassSVM'], help='termination classifier to be used')
    parser.add_argument('feature_extractor', type=str, choices=['RawImage', 'DownsampleImage'], help='feature extractor to be used')

    args = parser.parse_args()

    trajs_generator = load_trajectories(args.filepath, skip=0)
    ram_trajs, frame_trajs = parse_trajectories(trajs_generator, start=500, end=700)

    # (player_x, player_y) of good subgoals
    # [right plat, bottom of ladder of right plat, bottom of ladder of left plat,
    #  top of ladder of left plat, key, door]
    subgoals = [(133, 192), (132, 148), (20, 148), (20, 192), (13, 198), (21, 235)]
    subgoal = subgoals[0]

    traj_idx, state_idx = find_first_instance(ram_trajs, subgoal)
    window_sz = 5

    subgoal_ram = ram_trajs[traj_idx][state_idx]
    ground_truth_idxs = filter_in_term_set(ram_trajs, subgoal_ram)

    if args.feature_extractor == 'RawImage':
        feature_extractor = RawImage()
    elif args.feature_extractor == 'DownsampleImage':
        feature_extractor = DownsampleImage()

    term_set_frames = frame_trajs[traj_idx][state_idx - window_sz: state_idx + window_sz]
    if args.term_classifier == 'OneClassSVM':
        term_classifier = OneClassSVMClassifier(term_set_frames, feature_extractor)

    output = set()
    for i, frame_traj in enumerate(frame_trajs):
        for j, state in enumerate(frame_traj):
            if term_classifier.is_term(state):
                output.add((i, j))

    ground_truth_idxs_set = set(ground_truth_idxs)
    true_pos = len(ground_truth_idxs_set.intersection(output))
    false_pos = len(output) - true_pos
    print(f"Number of states in term set: {len(ground_truth_idxs_set)}")
    print(f"Number of true positives: {true_pos}")
    print(f"Number of false positives: {false_pos}")
