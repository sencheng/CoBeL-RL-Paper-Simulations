# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load results
activity_right_latent = np.array(pickle.load(open('data/results_latent.pkl', 'rb'))['activity_right'])
# ensure that the directory for plotting the successor representation exists
os.makedirs('plots/', exist_ok=True)

# plot successor representation of the starting state (#24)
font = {'size': 20}
plt.rc('font', **font)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))
# successor representation of the starting state (#24) at the end of the training phase 
sr_end_training = np.mean(activity_right_latent[:, -3, :, 24], axis=0)
# successor representation of the starting state (#24) at the end the simulation
sr_end_simulation = np.mean(activity_right_latent[:, -1, :, 24], axis=0)
ax1 = sns.heatmap(sr_end_training.reshape(5, 8), cmap=plt.get_cmap('binary'), linewidth=0.5, cbar_kws={'label': 'SR'}, ax=ax1)
ax2 = sns.heatmap(sr_end_simulation.reshape(5, 8), cmap=plt.get_cmap('binary'), linewidth=0.5, cbar_kws={'label': 'SR'}, ax=ax2)
# removing tick labels
for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig('plots/successor_representation.svg', dpi=200, bbox_inches='tight', transparent=True)
fig.savefig('plots/successor_representation.png', dpi=200, bbox_inches='tight', transparent=True)
