import csv

def save_results(args, window_sz, nu, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1, filepath):
    with open(filepath, "a") as f:
        writer = csv.writer(f)
        if args.label_extractor == 'OracleExtractor':
            if args.term_classifier == 'OneClassSVM':
                writer.writerow([nu, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
            elif args.term_classifier == 'TwoClassSVM':
                writer.writerow([gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
        else:
            if args.term_classifier == 'OneClassSVM':
                writer.writerow([window_sz, nu, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
            elif args.term_classifier == 'TwoClassSVM':
                writer.writerow([window_sz, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])

def save_results_sift(args, num_keypoints, num_clusters, window_sz, nu, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1, filepath):
    with open(filepath, "a") as f:
        writer = csv.writer(f)
        if args.label_extractor == 'OracleExtractor':
            if args.term_classifier == 'OneClassSVM':
                writer.writerow([num_keypoints, num_clusters, nu, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
            elif args.term_classifier == 'TwoClassSVM':
                writer.writerow([num_keypoints, num_clusters, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
        else:
            if args.term_classifier == 'OneClassSVM':
                writer.writerow([num_keypoints, num_clusters, window_sz, nu, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
            elif args.term_classifier == 'TwoClassSVM':
                writer.writerow([num_keypoints, num_clusters, window_sz, gamma, true_pos, false_pos, ground_truth_num, precision, recall, f1])
