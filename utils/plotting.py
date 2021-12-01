import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

def plot_SVM(classifier, ram_xy_states, states, is_xy, filepath):
   # extract the model predictions
   predicted = np.array([classifier.predict(state) for state in states])

   # define the meshgrid
   x_min, x_max = ram_xy_states[:, 0].min() - 5, ram_xy_states[:, 0].max() + 5
   y_min, y_max = ram_xy_states[:, 1].min() - 5, ram_xy_states[:, 1].max() + 5

   xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                        np.arange(y_min, y_max, 0.2))

   if is_xy:
      # evaluate the decision function on the meshgrid
      z = classifier.term_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
      z = z.reshape(xx.shape)

      # plot the decision function
      plt.contourf(xx, yy, z, cmap=plt.cm.PuBu)
      a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='darkred')

   # plot predictions
   pos = predicted == 1
   neg = np.logical_or(predicted == 0, predicted == -1) # TransductiveExtractor has 2 classes for neg
   c = plt.scatter(ram_xy_states[neg, 0], ram_xy_states[neg, 1], c='gold', edgecolors='k')
   b = plt.scatter(ram_xy_states[pos, 0], ram_xy_states[pos, 1], c='red', edgecolors='k')

   if is_xy:
      plt.legend([a.collections[0], b, c], ['learned frontier', 'regular observations', 'abnormal observations'], bbox_to_anchor=(1.05, 1))
   else:
      plt.legend([b, c], ['regular observations', 'abnormal observations'], bbox_to_anchor=(1.05, 1))

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
