# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load the simulation data
    results = pickle.load(open('data/rewards.pkl', 'rb'))
    
    # ensure that the plots directiory exists
    os.makedirs('plots/', exist_ok=True)
    
    # plot results
    plt.figure(1)
    plt.plot(np.arange(results.shape[1]) + 1, np.mean(results, axis=0), color='g')
    plt.axhline(0.5, linestyle='--', color='grey')
    plt.xlabel('Trial')
    plt.ylabel('Average Reward')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(1, 50)
    plt.ylim(0, 1.1)
    plt.xticks(np.array([1, 10, 20, 30, 40, 50]), np.array([1, 10, 20, 30, 40, 50]))
    plt.savefig('plots/rewards.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/rewards.svg', dpi=200, bbox_inches='tight', transparent=True)
