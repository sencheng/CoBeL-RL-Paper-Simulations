# basic imports
import os
import pickle
import matplotlib.pyplot as plt


# load recorded activity
activity = pickle.load(open('data/activity.pkl', 'rb'))

# plot activity for different trials
trials = [30, 70, 100, 130, 170, 199]
for i in range(6):
    plt.figure(i+1, figsize=(4, 3))
    plt.pcolor(activity[trials[i]][:60, :], cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('Observation')
    plt.ylabel('Unit')
    plt.xticks([])
    plt.yticks([])
    os.makedirs('plots/', exist_ok=True)
    plt.savefig('plots/activity_' + str(i) + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/activity_' + str(i) + '.svg', dpi=200, bbox_inches='tight', transparent=True)
