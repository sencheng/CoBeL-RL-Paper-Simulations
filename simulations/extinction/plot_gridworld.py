# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_occupancy(trajectories: list) -> np.ndarray:
    '''
    This function computes the occupancy map for the last 10 trials of the acquistion and extinction phases.
    
    Parameters
    ----------
    escape_latency :                    A list containing the trajectories collected during different simulation runs.
    
    Returns
    ----------
    occupancy :                         The occupancy map.
    '''
    O = np.zeros((2, 7, 7))
    for run in trajectories:
        # end of acquistion
        for trajectory in run[90:100]:
            for position in trajectory:
                y, x = np.array(position).astype(int)
                O[0, x, y] += 1.
        # end of extinction
        for trajectory in run[190:]:
            for position in trajectory:
                y, x = np.array(position).astype(int)
                O[1, x, y] += 1.
    O[0] /= np.sum(O[0])
    O[1] /= np.sum(O[1])
            
    return O


# load the simulation data
results = pickle.load(open('data/results_gridworld.pkl', 'rb'))

# compute escape latency stats
el_mean = np.mean(np.array(results['escape_latency']), axis=0)
el_std = np.std(np.array(results['escape_latency']), axis=0, ddof=1)
el_min = np.amin(np.array(results['escape_latency']), axis=0)
el_max = np.amax(np.array(results['escape_latency']), axis=0)

# compute CRC stats
for run in range(len(results['responses'])):
    for trial in range(1, len(results['responses'][run])):
        results['responses'][run][trial] += results['responses'][run][trial - 1]        
CRC_mean = np.mean(np.array(results['responses']), axis=0)
CRC_std = np.std(np.array(results['responses']), axis=0, ddof=1)

# compute occupancy map for the last 10 trials of the acquisition phase
occupancy = compute_occupancy(results['trajectory'])

# ensure that the plots directiory exists
os.makedirs('plots/', exist_ok=True)

# plot escape latency and CRC
plt.figure(1, figsize=(8, 2))
plt.subplots_adjust(wspace=0.35)
plt.subplot(1, 2, 1)
plt.xlabel('Trial')
plt.ylabel('Escape Latency [#steps]')
plt.xlim(1, 200)
plt.ylim(8, 40)
plt.plot(np.arange(200) + 1, el_mean, color='r')
plt.fill_between(np.arange(200) + 1, el_mean + el_std, el_mean - el_std, facecolor='r', alpha=0.5)
plt.axvline(100, linestyle='--', color='gray')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.subplot(1, 2, 2)
plt.xlabel('Trial')
plt.ylabel('Cumulative Response')
plt.xlim(1, 200)
plt.ylim(0, 110)
plt.plot(np.arange(200) + 1, CRC_mean, color='b')
plt.fill_between(np.arange(200) + 1, CRC_mean + CRC_std, CRC_mean - CRC_std, facecolor='b', alpha=0.5)
plt.axvline(100, linestyle='--', color='gray')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('plots/gridworld_extinction.png', dpi=200, bbox_inches='tight')
plt.savefig('plots/gridworld_extinction.svg', dpi=200, bbox_inches='tight')

# plot occupancy map
plt.figure(2, figsize=(15, 6))
for i, o in enumerate(occupancy):
    plt.subplot(1, 2, i+1)
    plt.pcolor(o, cmap='hot', vmin=0)
    plt.colorbar()
    plt.plot(np.array([3, 3]), np.array([1, 6]), color='cyan', linewidth=5)
    plt.plot(np.array([4, 4]), np.array([1, 6]), color='cyan', linewidth=5)
    plt.plot(np.array([0, 3]), np.array([6, 6]), color='cyan', linewidth=5)
    plt.plot(np.array([4, 7]), np.array([6, 6]), color='cyan', linewidth=5)
    plt.plot(np.array([3, 4]), np.array([1, 1]), color='cyan', linewidth=5)
    plt.plot(np.array([0, 7]), np.array([7, 7]), color='cyan', linewidth=10)
    plt.plot(np.array([0, 0]), np.array([6.1, 7]), color='cyan', linewidth=10)
    plt.plot(np.array([7, 7]), np.array([6.1, 7]), color='cyan', linewidth=10)
    plt.xticks([])
    plt.yticks([])
plt.savefig('plots/gridworld_extinction_occupancy.png', dpi=200, bbox_inches='tight')
plt.savefig('plots/gridworld_extinction_occupancy.svg', dpi=200, bbox_inches='tight')
