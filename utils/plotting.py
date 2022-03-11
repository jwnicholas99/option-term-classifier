import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

from utils.monte_preprocessing import parse_ram_xy

def plot_SVM(classifier, trajs, raw_ram_trajs, ground_truth_idxs_set, filepath):
   # extract the model predictions
   ground_truth_states = []
   ground_truth_ram_xy_states = []
   non_ground_truth_states = []
   non_ground_truth_ram_xy_states = []

   for traj_idx, traj in enumerate(trajs):
       for state_idx, state in enumerate(traj):
           if (traj_idx, state_idx) in ground_truth_idxs_set:
               ground_truth_states.append(state)
               ground_truth_ram_xy_states.append(parse_ram_xy(raw_ram_trajs[traj_idx][state_idx]))
           else:
               non_ground_truth_states.append(state)
               non_ground_truth_ram_xy_states.append(parse_ram_xy(raw_ram_trajs[traj_idx][state_idx]))

   ground_truth_states = np.array(ground_truth_states)
   ground_truth_ram_xy_states = np.array(ground_truth_ram_xy_states)
   non_ground_truth_states = np.array(non_ground_truth_states)
   non_ground_truth_ram_xy_states = np.array(non_ground_truth_ram_xy_states)

   if len(ground_truth_states) == 0 or len(non_ground_truth_states) == 0:
      print("No ground truth states or non ground truth states to plot")
      return

   ground_truth_preds = np.array(classifier.predict(ground_truth_states))
   non_ground_truth_preds = np.array(classifier.predict(non_ground_truth_states))

   '''
   Plot predictions - there are 4 possible types: 
      1. True Positive: States that are predicted in term set AND is in ground truth set (red +)
      2. False Positive: States that are predicted in term set AND is not in ground truth set (red o)
      3. True Negative: States that are predicted not in term set AND is not in ground truth set (gold o)
      4. False Negative: States that are predicted not in term set AND is in ground truth set (gold +)
   '''
   true_pos = plt.scatter(ground_truth_ram_xy_states[ground_truth_preds == 1, 0], 
                          ground_truth_ram_xy_states[ground_truth_preds == 1, 1],
                          c='red', edgecolors='k', marker="P")
   false_pos = plt.scatter(non_ground_truth_ram_xy_states[non_ground_truth_preds == 1, 0], 
                           non_ground_truth_ram_xy_states[non_ground_truth_preds == 1, 1],
                           c='red', edgecolors='k')
   true_neg = plt.scatter(non_ground_truth_ram_xy_states[non_ground_truth_preds != 1, 0], 
                          non_ground_truth_ram_xy_states[non_ground_truth_preds != 1, 1],
                          c='gold', edgecolors='k')
   false_neg = plt.scatter(ground_truth_ram_xy_states[ground_truth_preds != 1, 0], 
                           ground_truth_ram_xy_states[ground_truth_preds != 1, 1],
                           c='gold', edgecolors='k', marker="P")

   plt.legend([true_pos, false_pos, true_neg, false_neg], ['true pos', 'false pos', 'true neg', 'false neg'], bbox_to_anchor=(1.05, 1))

   plt.axis('tight')
   plt.savefig(filepath)
   plt.clf()

def plot_hyperparam_search(src_path, dest_path):
   with open(src_path) as f:
      csv_reader = csv.reader(f, delimiter=',')
      x, y, z, true_pos, false_pos = [], [], [], [], []
      for row in csv_reader:
         x.append(float(row[0]))
         y.append(float(row[1]))
         if row[2] == 'scale':
            z.append(1.0)
         elif row[2] == 'auto':
            z.append(2.0)
         else:
            z.append(float(row[2]))
         true_pos.append(float(row[3]))
         false_pos.append(float(row[4]))

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      #img = ax.scatter(x, y, z, c=c, cmap=plt.autumn())
      img = ax.scatter(y, z, true_pos, color='blue', label='true_pos')
      img = ax.scatter(y, z, false_pos, color='red', label='false_pos')
      ax.set_xlabel('nu')
      ax.set_ylabel('gamma')
      ax.set_zlabel('number of states')
      ax.legend()
      #fig.colorbar(img)
      plt.savefig(dest_path)
      plt.clf()
