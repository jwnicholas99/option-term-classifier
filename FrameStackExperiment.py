import numpy as np
import torch
import copy

from classifiers.OneClassSVM import OneClassSVMClassifier
from classifiers.TwoClassSVM import TwoClassSVMClassifier
from classifiers.FullCNN import FullCNN

from feature_extractors.RawImage import RawImage
from feature_extractors.DownsampleImage import DownsampleImage
from feature_extractors.RawRAM import RawRAM
from feature_extractors.MonteRAMState import MonteRAMState
from feature_extractors.MonteRAMXY import MonteRAMXY
from feature_extractors.BOVW_stack import BOVW_stack
from feature_extractors.RND import RND
from feature_extractors.CNN import CNN
from feature_extractors.CNN_stack import CNN_stack

from label_extractors.OracleExtractor import OracleExtractor
from label_extractors.BeforeAfterExtractor import BeforeAfterExtractor
from label_extractors.AfterExtractor import AfterExtractor
from label_extractors.TransductiveExtractor import TransductiveExtractor
from label_extractors.PositiveAugmentExtractor import PositiveAugmentExtractor
from label_extractors.labeling_funcs import square_epsilon, square_epsilon_screen

from utils.statistics import calc_statistics
from utils.save_results import save_results, save_results_sift
from utils.plotting import plot_SVM
from utils.monte_preprocessing import parse_ram_xy

class FrameStackExperiment():
    def __init__(self, train_trajs, train_raw_ram_trajs, test_trajs, test_raw_ram_trajs,
                 subgoals, subgoals_info, 
                 args, hyperparams):
        self.train_trajs = [self.construct_frame_stacks(traj) for traj in train_trajs]
        self.train_raw_ram_trajs = train_raw_ram_trajs
        self.test_trajs = [self.construct_frame_stacks(traj) for traj in test_trajs]
        self.test_raw_ram_trajs = test_raw_ram_trajs

        self.subgoals = subgoals
        self.subgoals_info = subgoals_info

        self.args = args
        self.hyperparams = hyperparams

    def construct_frame_stacks(self, traj):
        frame_stacks = []
        for idx in range(len(traj)):
            frame_stack = [traj[min(0, idx-3)], traj[min(0, idx-2)], traj[min(0, idx-1)], traj[idx]]
            frame_stacks.append(frame_stack)
        return frame_stacks

    def run(self):
        total_true_pos = 0
        total_false_pos = 0
        total_ground_truth = 0
        f1_scores = []

        num_sift_keypoints = self.hyperparams["num_sift_keypoints"]
        num_clusters = self.hyperparams["num_clusters"]
        window_sz = self.hyperparams["window_sz"]
        nu = self.hyperparams["nu"]
        gamma = self.hyperparams["gamma"]

        # Create frame stacker utils
        if self.args.feature_extractor == 'BOVW':
            train_start = 0
            train_end = 35
            #train_end = 5

            train_states = [state for traj in self.train_trajs[train_start:train_end] for state in traj]
            print(len(train_states))

            feature_extractor = BOVW_stack(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)
            feature_extractor.train(train_states)
        elif self.args.feature_extractor == 'CNN':
            feature_extractor = CNN_stack()

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
            elif self.args.feature_extractor == 'RND':
                feature_extractor = RND('MontezumaRevengeNoFrameskip-v4.pred')

            # Set-up label extractor
            if self.args.label_extractor == 'BeforeAfterExtractor':
                label_extractor = BeforeAfterExtractor(self.args.extract_only_pos, window_sz)
            elif self.args.label_extractor == 'AfterExtractor':
                label_extractor = AfterExtractor(self.args.extract_only_pos, window_sz)
            elif self.args.label_extractor == 'OracleExtractor':
                label_extractor = OracleExtractor(square_epsilon_screen, self.args.extract_only_pos)
            elif self.args.label_extractor == 'TransductiveExtractor':
                label_extractor = TransductiveExtractor(self.args.extract_only_pos, window_sz)
            elif self.args.label_extractor == 'PositiveAugmentExtractor':
                label_extractor = PositiveAugmentExtractor(feature_extractor, self.args.extract_only_pos, window_sz)

            # Set-up classifier
            if self.args.term_classifier == 'OneClassSVM':
                classifier = OneClassSVMClassifier(feature_extractor, window_sz=window_sz, nu=nu, gamma=gamma)
            elif self.args.term_classifier == 'TwoClassSVM':
                classifier = TwoClassSVMClassifier(feature_extractor, window_sz=window_sz, gamma=gamma)
            elif self.args.term_classifier == 'FullCNN':
                num_classes = 2
                if self.args.label_extractor == 'TransductiveExtractor':
                    num_classes = 3
                elif self.args.label_extractor == 'PositiveAugmentExtractor':
                    num_classes = 4
                classifier = FullCNN('cuda:0' if torch.cuda.is_available() else 'cpu', n_classes=num_classes)
            train_data, labels = label_extractor.extract_labels(self.train_trajs, self.train_raw_ram_trajs, traj_idx, state_idx)
            classifier.train(train_data, labels)

            # Different tests for frame stacks
            # 1. Every frame stack in the training data should be classified as pos
            '''
            pos_count = 0
            pos_train_data = [train_data[i] for i in range(len(labels)) if labels[i]]
            for frame_stack in pos_train_data:
                if classifier.predict([frame_stack])[0]:
                    pos_count += 1
            print("train_data pos: ", len(pos_train_data))
            print(pos_count)

            # 2. For each frame stack in the training data, flip the order of the frames - should classify as neg
            neg_count = 0
            for frame_stack in pos_train_data:
                reversed_stack = frame_stack[::-1]
                if not classifier.predict([reversed_stack])[0]:
                    neg_count += 1
            print(neg_count)

            # 3. For each frame stack in the training data, change one frame to be some other part of the room - should classify as neg
            # I'm using the starting position for now as it should not be in any term sets
            neg_count = 0
            starting_frame = self.train_trajs[0][0][0]
            for frame_stack in pos_train_data:
                modified_stack = frame_stack[0:]
                modified_stack[1] = starting_frame
                modified_stack[2] = starting_frame
                if not classifier.predict([modified_stack])[0]:
                    neg_count += 1
            print(neg_count)
            '''

            # Plot trained classifier on test set
            ground_truth_idxs_set = set(ground_truth_idxs)
            if self.args.term_classifier == 'OneClassSVM':
                file_path = f"{self.args.dest}/plots/test/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_nu={nu}_gamma={gamma}.png"
            elif self.args.term_classifier == 'TwoClassSVM':
                file_path = f"{self.args.dest}/plots/test/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_gamma={gamma}.png"
            plot_SVM(classifier, self.test_trajs, self.test_raw_ram_trajs, ground_truth_idxs_set, file_path)

            ground_truth_idxs_set = set()
            if self.args.term_classifier == 'OneClassSVM':
                file_path = f"{self.args.dest}/plots/train/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_nu={nu}_gamma={gamma}.png"
            elif self.args.term_classifier == 'TwoClassSVM':
                file_path = f"{self.args.dest}/plots/train/sift={num_sift_keypoints}_clusters={num_clusters}_x={subgoal[0]}_y={subgoal[1]}_windowsz={window_sz}_gamma={gamma}.png"
            train_ram, labels = label_extractor.extract_labels(self.train_raw_ram_trajs, self.train_raw_ram_trajs, traj_idx, state_idx)
            plot_SVM(classifier, train_data, train_ram, ground_truth_idxs_set, file_path)
