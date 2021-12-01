import argparse
import gzip
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classifiers.OneClassSVM import OneClassSVMClassifier
from classifiers.TwoClassSVM import TwoClassSVMClassifier

from feature_extractors.RawImage import RawImage
from feature_extractors.DownsampleImage import DownsampleImage
from feature_extractors.RawRAM import RawRAM
from feature_extractors.MonteRAMState import MonteRAMState
from feature_extractors.MonteRAMXY import MonteRAMXY

from label_extractors.OracleExtractor import OracleExtractor
from label_extractors.BeforeAfterExtractor import BeforeAfterExtractor
from label_extractors.AfterExtractor import AfterExtractor
from label_extractors.TransductiveExtractor import TransductiveExtractor
from label_extractors.labeling_funcs import square_epsilon

from utils.monte_preprocessing import parse_ram, parse_ram_xy
from utils.plotting import plot_SVM
from utils.statistics import calc_statistics

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
        (List of lists of raw RAM, List of lists of MonteRAMState, List of lists of frames)
    '''
    raw_ram_trajs, ram_trajs, frame_trajs = [], [], []
    for i, traj in enumerate(trajs):
        if i < start:
            continue
        if i > end:
            break

        raw_ram_traj, ram_traj, frame_traj = [], [], []
        for ram, frame in traj:
            raw_ram_traj.append(ram)
            ram_traj.append(parse_ram(ram))
            frame_traj.append(frame)
        raw_ram_trajs.append(raw_ram_traj)
        ram_trajs.append(ram_traj)
        frame_trajs.append(frame_traj)
    return raw_ram_trajs, ram_trajs, frame_trajs

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
        for j, state in enumerate(traj):
            if square_epsilon(subgoal, state):
                term_set.append((i, j))
    return term_set

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate termination classifier performance')

    parser.add_argument('filepath', type=str, help='filepath of pkl file containing trajectories with RAM states and frames')
    parser.add_argument('term_classifier', type=str, choices=['OneClassSVM', 'TwoClassSVM'], help='termination classifier to be used')
    parser.add_argument('feature_extractor', type=str, choices=['RawImage', 'DownsampleImage', 'RawRAM', 'MonteRAMState', 'MonteRAMXY'], help='feature extractor to be used')
    parser.add_argument('label_extractor', type=str, choices=['BeforeAfterExtractor', 'AfterExtractor', 'OracleExtractor', 'TransductiveExtractor'], help='label extractor to be used')
    parser.add_argument('--extract_only_pos', default=False, action='store_true', help='whether label extractor should only extract positive egs')

    args = parser.parse_args()

    trajs_generator = load_trajectories(args.filepath, skip=0)
    raw_ram_trajs, ram_trajs, frame_trajs = parse_trajectories(trajs_generator, start=500, end=600)

    # (player_x, player_y) of good subgoals
    # [right plat, bottom of ladder of right plat, bottom of ladder of left plat,
    #  top of ladder of left plat, key, door]
    subgoals = [(133, 192), (132, 148), (20, 148), (20, 192), (13, 198), (21, 235)]
    subgoal = subgoals[0]

    for window_sz in range(0, 7):
    #for window_sz in range(0, 1):
        #for nu in np.arange(0.1, 0.5, 0.1):
        for nu in np.arange(0.1, 0.2, 0.1):
            for gamma in [0.0001, 0.001, 0.01, 0.1, 'scale', 'auto']:
                print(f"[+] Running with window_sz={window_sz}, nu={nu}, gamma={gamma}")

                traj_idx, state_idx = find_first_instance(ram_trajs, subgoal)

                subgoal_ram = ram_trajs[traj_idx][state_idx]
                ground_truth_idxs = filter_in_term_set(ram_trajs, subgoal_ram)

                # Set-up feature extractor
                if args.feature_extractor == 'RawImage':
                    feature_extractor = RawImage()
                elif args.feature_extractor == 'DownsampleImage':
                    feature_extractor = DownsampleImage()
                elif args.feature_extractor == 'RawRAM':
                    feature_extractor = RawRAM()
                elif args.feature_extractor == 'MonteRAMState':
                    feature_extractor = MonteRAMState()
                elif args.feature_extractor == 'MonteRAMXY':
                    feature_extractor = MonteRAMXY()

                # Extract positive and negative labels
                if args.label_extractor == 'BeforeAfterExtractor':
                    label_extractor = BeforeAfterExtractor(args.extract_only_pos, window_sz)
                elif args.label_extractor == 'AfterExtractor':
                    label_extractor = AfterExtractor(args.extract_only_pos, window_sz)
                elif args.label_extractor == 'OracleExtractor':
                    label_extractor = OracleExtractor(square_epsilon, args.extract_only_pos)
                elif args.label_extractor == 'TransductiveExtractor':
                    label_extractor = TransductiveExtractor(args.extract_only_pos, window_sz)

                if args.feature_extractor == 'RawImage' or args.feature_extractor == 'DownsampleImage':
                    subgoal_traj = frame_trajs[traj_idx]
                    trajs = frame_trajs
                elif args.feature_extractor == 'RawRAM' or args.feature_extractor == 'MonteRAMState' or args.feature_extractor == 'MonteRAMXY':
                    subgoal_traj = raw_ram_trajs[traj_idx]
                    trajs = raw_ram_trajs

                train_data, labels = label_extractor.extract_labels(trajs, traj_idx, state_idx)

                # Set-up classifier
                if args.term_classifier == 'OneClassSVM':
                    term_classifier = OneClassSVMClassifier(feature_extractor, window_sz=window_sz, nu=nu, gamma=gamma)
                elif args.term_classifier == 'TwoClassSVM':
                    term_classifier = TwoClassSVMClassifier(feature_extractor, window_sz=window_sz, gamma=gamma)
                term_classifier.train(train_data, labels)

                # Evaluate classifier
                output = set()
                for i, traj in enumerate(trajs):
                    for j, state in enumerate(traj):
                        if term_classifier.predict(state):
                            output.add((i, j))

                # Plot trained classifier
                all_states = np.array([state for traj in trajs for state in traj])
                ram_xy_states = np.array([parse_ram_xy(state) for traj in raw_ram_trajs for state in traj])
                is_xy = args.feature_extractor == 'MonteRAMXY'

                if args.term_classifier == 'OneClassSVM':
                    file_path = f"{args.label_extractor}_plots/all_states_windowsz={window_sz}_nu={nu}_gamma={gamma}.png"
                elif args.term_classifier == 'TwoClassSVM':
                    file_path = f"{args.label_extractor}_plots/all_states_windowsz={window_sz}_gamma={gamma}.png"
                plot_SVM(term_classifier, ram_xy_states, all_states, is_xy, file_path)

                # Calculate and save statistics
                ground_truth_idxs_set = set(ground_truth_idxs)
                true_pos = len(ground_truth_idxs_set.intersection(output))
                false_pos = len(output) - true_pos
                precision, recall, f1 = calc_statistics(true_pos, false_pos, len(ground_truth_idxs_set))
                
                print(f"Number of states in term set: {len(ground_truth_idxs_set)}")
                print(f"Number of true positives: {true_pos}")
                print(f"Number of false positives: {false_pos}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1: {f1}")

                with open(f"{args.label_extractor}_results.csv", "a") as f:
                    writer = csv.writer(f)
                    if args.label_extractor == 'OracleExtractor':
                        if args.term_classifier == 'OneClassSVM':
                            writer.writerow([nu, gamma, true_pos, false_pos, precision, recall, f1])
                        elif args.term_classifier == 'TwoClassSVM':
                            writer.writerow([gamma, true_pos, false_pos, precision, recall, f1])
                    else:
                        if args.term_classifier == 'OneClassSVM':
                            writer.writerow([window_sz, nu, gamma, true_pos, false_pos, precision, recall, f1])
                        elif args.term_classifier == 'TwoClassSVM':
                            writer.writerow([window_sz, gamma, true_pos, false_pos, precision, recall, f1])
