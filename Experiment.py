import numpy as np

from classifiers.OneClassSVM import OneClassSVMClassifier
from classifiers.TwoClassSVM import TwoClassSVMClassifier

from feature_extractors.RawImage import RawImage
from feature_extractors.DownsampleImage import DownsampleImage
from feature_extractors.RawRAM import RawRAM
from feature_extractors.MonteRAMState import MonteRAMState
from feature_extractors.MonteRAMXY import MonteRAMXY
from feature_extractors.BOVW import BOVW

from label_extractors.OracleExtractor import OracleExtractor
from label_extractors.BeforeAfterExtractor import BeforeAfterExtractor
from label_extractors.AfterExtractor import AfterExtractor
from label_extractors.TransductiveExtractor import TransductiveExtractor
from label_extractors.labeling_funcs import square_epsilon, square_epsilon_screen

from utils.statistics import calc_statistics
from utils.save_results import save_results, save_results_sift
from utils.plotting import plot_SVM
from utils.monte_preprocessing import parse_ram_xy

class Experiment():
    def __init__(self, train_trajs, train_raw_ram_trajs, test_trajs, test_raw_ram_trajs,
                 subgoals, subgoals_info, 
                 args, hyperparams):
        self.train_trajs = train_trajs
        self.train_raw_ram_trajs = train_raw_ram_trajs
        self.test_trajs = test_trajs
        self.test_raw_ram_trajs = test_raw_ram_trajs

        self.subgoals = subgoals
        self.subgoals_info = subgoals_info

        self.args = args
        self.hyperparams = hyperparams

    def run(self):
        total_true_pos = 0
        total_false_pos = 0
        total_ground_truth = 0

        num_sift_keypoints = self.hyperparams["num_sift_keypoints"]
        num_clusters = self.hyperparams["num_clusters"]
        window_sz = self.hyperparams["window_sz"]
        nu = self.hyperparams["nu"]
        gamma = self.hyperparams["gamma"]

        # Train feature extractor
        if self.args.feature_extractor == 'BOVW':
            train_start = 0
            train_end = 35

            train_states = [state for traj in self.train_trajs[train_start:train_end] for state in traj]

            feature_extractor = BOVW(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)
            feature_extractor.train(train_states)
            feature_extractor.visualize_sift_feats([self.train_trajs[0][0]], f"{self.args.dest}/sifts/")

        for subgoal in self.subgoals:
            print(f"[+] Subgoal: {subgoal}")
            if subgoal not in self.subgoals_info:
                print(f"No trajectories for {subgoal}")
                continue

            subgoal_info = self.subgoals_info[subgoal]
            traj_idx = subgoal_info['traj_idx']
            state_idx = subgoal_info['state_idx']
            ground_truth_idxs = subgoal_info['ground_truth_idxs']

            # Set-up feature extractor
            if self.args.feature_extractor == 'RawImage':
                feature_extractor = RawImage()
            elif self.args.feature_extractor == 'DownsampleImage':
                feature_extractor = DownsampleImage()
            elif self.args.feature_extractor == 'RawRAM':
                feature_extractor = RawRAM()
            elif self.args.feature_extractor == 'MonteRAMState':
                feature_extractor = MonteRAMState()
            elif self.args.feature_extractor == 'MonteRAMXY':
                feature_extractor = MonteRAMXY()

            # Set-up label extractor
            if self.args.label_extractor == 'BeforeAfterExtractor':
                label_extractor = BeforeAfterExtractor(self.args.extract_only_pos, window_sz)
            elif self.args.label_extractor == 'AfterExtractor':
                label_extractor = AfterExtractor(self.args.extract_only_pos, window_sz)
            elif self.args.label_extractor == 'OracleExtractor':
                label_extractor = OracleExtractor(square_epsilon_screen, self.args.extract_only_pos)
            elif self.args.label_extractor == 'TransductiveExtractor':
                label_extractor = TransductiveExtractor(self.args.extract_only_pos, window_sz)

            # Set-up classifier
            if self.args.term_classifier == 'OneClassSVM':
                classifier = OneClassSVMClassifier(feature_extractor, window_sz=window_sz, nu=nu, gamma=gamma)
            elif self.args.term_classifier == 'TwoClassSVM':
                classifier = TwoClassSVMClassifier(feature_extractor, window_sz=window_sz, gamma=gamma)
            train_data, labels = label_extractor.extract_labels(self.train_trajs, self.train_raw_ram_trajs, traj_idx, state_idx)
            classifier.train(train_data, labels)

            output = set()
            for i, test_traj in enumerate(self.test_trajs):
                for j, test_state in enumerate(test_traj):
                    if classifier.predict([test_state])[0]:
                        output.add((i, j))

            # Calculate and save statistics
            ground_truth_idxs_set = set(ground_truth_idxs)
            ground_truth_num = len(ground_truth_idxs_set)
            true_pos = len(ground_truth_idxs_set.intersection(output))
            false_pos = len(output) - true_pos

            total_true_pos += true_pos
            total_false_pos += false_pos
            total_ground_truth += ground_truth_num

            precision, recall, f1 = calc_statistics(true_pos, false_pos, ground_truth_num)
            
            print(f"Number of states in term set: {ground_truth_num}")
            print(f"Number of true positives: {true_pos}")
            print(f"Number of false positives: {false_pos}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1: {f1}")

            save_results_sift(self.args, num_sift_keypoints, num_clusters, window_sz, nu, gamma, 
                              true_pos, false_pos, ground_truth_num,
                              precision, recall, f1,
                              f"{self.args.dest}/{subgoal[0]}_{subgoal[1]}_results.csv")

            # Plot trained classifier on train set
            all_states = np.array([state for traj in self.train_trajs for state in traj])
            ram_xy_states = np.array([parse_ram_xy(state) for traj in self.train_raw_ram_trajs for state in traj])
            is_xy = self.args.feature_extractor == 'MonteRAMXY'

            if self.args.term_classifier == 'OneClassSVM':
                file_path = f"{self.args.dest}/plots/train/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_nu={nu}_gamma={gamma}.png"
            elif self.args.term_classifier == 'TwoClassSVM':
                file_path = f"{self.args.dest}/plots/train/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_gamma={gamma}.png"
            plot_SVM(classifier, ram_xy_states, all_states, is_xy, file_path)

            # Plot trained classifier on test set
            all_states = np.array([state for traj in self.test_trajs for state in traj])
            ram_xy_states = np.array([parse_ram_xy(state) for traj in self.test_raw_ram_trajs for state in traj])

            if self.args.term_classifier == 'OneClassSVM':
                file_path = f"{self.args.dest}/plots/test/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_nu={nu}_gamma={gamma}.png"
            elif self.args.term_classifier == 'TwoClassSVM':
                file_path = f"{self.args.dest}/plots/test/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_gamma={gamma}.png"
            plot_SVM(classifier, ram_xy_states, all_states, is_xy, file_path)

        # Calculate and save overall statistics across subgoals
        overall_precision, overall_recall, overall_f1 = calc_statistics(total_true_pos, total_false_pos, total_ground_truth)
        save_results_sift(self.args, num_sift_keypoints, num_clusters, window_sz, nu, gamma, 
                          total_true_pos, total_false_pos, total_ground_truth,
                          overall_precision, overall_recall, overall_f1,
                          f"{self.args.dest}/{self.args.label_extractor}_overall_results.csv")
