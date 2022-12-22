# basic imports
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# load recorded behavior
behavior = pickle.load(open('data/behavior.pkl', 'rb'))
for variable in ['reward', 'escape_latency', 'response']:
    behavior[variable] = np.array(behavior[variable])

# compute CRC
crc = np.copy(behavior['response'])
for i in range(crc.shape[1] - 1):
    crc[:, i+1] += crc[:, i]
    
# compute occupancy map
occupancy = np.zeros((2, 9, 9))
for run in behavior['trajectory']:
    # start
    for trial in run[:10]:
        for position in trial:
            y, x = position.astype(int)
            occupancy[0, x, y] += 1.
    # end
    for trial in run[-10:]:
        for position in trial:
            y, x = position.astype(int)
            occupancy[1, x, y] += 1.
occupancy[0] /= np.sum(occupancy[0])
occupancy[1] /= np.sum(occupancy[1])

# label font size
font_size=13

# make sure that the plots directory exists
os.makedirs('plots/', exist_ok=True)
    
# plot reward curve
plt.figure(1, figsize=(4, 3))
plt.plot(np.arange(behavior['reward'].shape[1]) + 1, np.mean(behavior['reward'], axis=0), color='g')
plt.xlabel('Trial', fontsize=font_size)
plt.ylabel('Trial Reward', fontsize=font_size)
plt.xlim(1, behavior['reward'].shape[1])
plt.ylim(0, 1.05)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('plots/reward.png', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/reward.svg', dpi=200, bbox_inches='tight', transparent=True)

# plot escape latency curve
plt.figure(2, figsize=(4, 3))
plt.plot(np.arange(behavior['escape_latency'].shape[1]) + 1, np.mean(behavior['escape_latency'], axis=0), color='r')
plt.xlabel('Trial', fontsize=font_size)
plt.ylabel('Escape Latency [#steps]', fontsize=font_size)
plt.xlim(1, behavior['escape_latency'].shape[1])
plt.ylim(8, 25)
plt.yticks(np.array([10, 15, 20, 25]))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('plots/escape_latency.png', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/escape_latency.svg', dpi=200, bbox_inches='tight', transparent=True)

# plot responses
plt.figure(3, figsize=(4, 3))
plt.plot(np.arange(behavior['response'].shape[1]) + 1, np.mean(behavior['response'], axis=0), color='b')
plt.xlabel('Trial', fontsize=font_size)
plt.ylabel('Conditioned Response', fontsize=font_size)
plt.xlim(1, behavior['response'].shape[1])
plt.ylim(0, 1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('plots/responses.png', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/responses.svg', dpi=200, bbox_inches='tight', transparent=True)

# plot CRC
plt.figure(4, figsize=(4, 3))
plt.plot(np.arange(crc.shape[1]) + 1, np.mean(crc, axis=0), color='b')
plt.xlabel('Trial', fontsize=font_size)
plt.ylabel('Cumulative Response', fontsize=font_size)
plt.xlim(1, crc.shape[1])
#plt.ylim(0, 1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('plots/crc.png', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/crc.svg', dpi=200, bbox_inches='tight', transparent=True)

# plot occupancy map
plt.figure(5, figsize=(11, 4))
for i, o in enumerate(occupancy):
    plt.subplot(1, 2, i+1)
    plt.pcolor(o, cmap='hot', vmin=0.)
    plt.colorbar()
    plt.plot(np.array([1, 8]), np.array([8, 8]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([1, 4]), np.array([7, 7]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([5, 8]), np.array([7, 7]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([4, 5]), np.array([2, 2]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([1, 1]), np.array([7, 8]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([4, 4]), np.array([2, 7]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([5, 5]), np.array([2, 7]), color='tab:cyan', linewidth=3)
    plt.plot(np.array([8, 8]), np.array([7, 8]), color='tab:cyan', linewidth=3)
    plt.xticks([])
    plt.yticks([])
plt.savefig('plots/occupancy.png', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/occupancy.svg', dpi=200, bbox_inches='tight', transparent=True)
