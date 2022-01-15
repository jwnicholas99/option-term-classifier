import argparse
import numpy as np

from Data import Data
from Experiment import Experiment

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate termination classifier performance')

    parser.add_argument('filepath', type=str, help='filepath of pkl file containing trajectories with RAM states and frames')
    parser.add_argument('dest', type=str, help='directory to write results and plots to')
    parser.add_argument('term_classifier', type=str, choices=['OneClassSVM', 'TwoClassSVM'], help='termination classifier to be used')
    parser.add_argument('feature_extractor', type=str, choices=['RawImage', 'DownsampleImage', 'RawRAM', 'MonteRAMState', 'MonteRAMXY', 'BOVW'], help='feature extractor to be used')
    parser.add_argument('label_extractor', type=str, choices=['BeforeAfterExtractor', 'AfterExtractor', 'OracleExtractor', 'TransductiveExtractor'], help='label extractor to be used')
    parser.add_argument('--extract_only_pos', default=False, action='store_true', help='whether label extractor should only extract positive egs')

    args = parser.parse_args()

    data = Data(args.filepath, train_skip=0, train_num=200, test_skip=0, test_num=100)

    # (player_x, player_y, screen) of good subgoals
    # [right plat, bottom of ladder of right plat, bottom of ladder of left plat,
    #  top of ladder of left plat, key, left door, right door]
    subgoals = [(133, 192, 1), (132, 148, 1), (20, 148, 1), (20, 192, 1), (13, 198, 1), (24, 235, 1), (130, 235, 1)]
    #subgoals = [(24, 235, 1), (130, 235, 1)]

    # Prepare hyperparams
    if args.label_extractor == 'OracleExtractor':
        window_sz_hyperparms = [None]
    else:
        window_sz_hyperparms = range(0, 7)

    if args.feature_extractor == 'BOVW':
        num_clusters_hyperparams = range(50, 100, 10)
        num_sift_keypoints_hyperparams = range(40, 70, 5)
    else:
        num_clusters_hyperparams = [None]
        num_sift_keypoints_hyperparams = [None]

    if args.term_classifier == 'OneClassSVM':
        nu_hyperparams = np.arange(0.1, 0.5, 0.1)
    else:
        nu_hyperparams = np.arange(0.1, 0.2, 0.1)

    gamma_hyperparams = [0.0001, 0.001, 0.01, 0.1, 'scale', 'auto']

    # Prepare information on each subgoal
    subgoals_info = {}
    for subgoal in subgoals:
        traj_idx, state_idx = data.find_first_instance(data.train_ram_trajs, subgoal)
        if traj_idx is None:
            continue

        subgoal_ram = data.train_ram_trajs[traj_idx][state_idx]
        ground_truth_idxs = data.filter_in_term_set(data.test_ram_trajs, subgoal_ram)
        subgoals_info[subgoal] = {'traj_idx': traj_idx,
                                  'state_idx': state_idx,
                                  'ground_truth_idxs': ground_truth_idxs}

    # Run experiments
    for num_clusters in num_clusters_hyperparams:
        for num_sift_keypoints in num_sift_keypoints_hyperparams:
            for window_sz in window_sz_hyperparms:
                for nu in nu_hyperparams:
                    for gamma in gamma_hyperparams:
                            print(f"[+] clusters={num_clusters}, kps={num_sift_keypoints}, window_sz={window_sz}, nu={nu}, gamma={gamma}")

                            if args.feature_extractor == 'RawImage' or args.feature_extractor == 'DownsampleImage' or args.feature_extractor == 'BOVW':
                                train_trajs = data.train_frame_trajs
                                test_trajs = data.test_frame_trajs
                            elif args.feature_extractor == 'RawRAM' or args.feature_extractor == 'MonteRAMState' or args.feature_extractor == 'MonteRAMXY':
                                train_trajs = data.train_raw_ram_trajs
                                test_trajs = data.test_raw_ram_trajs

                            # Run experiment
                            hyperparams = {
                                "num_sift_keypoints": num_sift_keypoints,
                                "num_clusters": num_clusters,
                                "window_sz": window_sz,
                                "nu": nu,
                                "gamma": gamma,
                            }
                            experiment = Experiment(train_trajs, data.train_raw_ram_trajs, test_trajs, data.test_raw_ram_trajs,
                                                    subgoals, subgoals_info, 
                                                    args, hyperparams)
                            experiment.run()

