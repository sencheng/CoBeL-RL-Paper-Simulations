# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# load results
el_latent = np.array(pickle.load(open('data/results_latent.pkl', 'rb'))['escape_latency'])
el_direct = np.array(pickle.load(open('data/results_direct.pkl', 'rb'))['escape_latency'])

# ensure that the directory for plotting the escape latency exists
os.makedirs('plots/', exist_ok=True)

# plot escape latency
range_latent, range_direct = range(-100, 100), range(0, 100)
font = {'size': 20}
plt.rc('font', **font)
plt.figure(1, figsize=(8.5, 5))
plt.plot(range_latent, np.mean(el_latent, axis=0), color='purple', linewidth=4, label='Latent Learning')
plt.plot(range_direct, np.mean(el_direct, axis=0), color='orange', linewidth=4, label='Direct Learning')
plt.xlabel('Trial')
plt.ylabel('Escape Latency')
plt.grid(axis='both', color='gray', linestyle='-', linewidth=0.5)
plt.legend(loc='upper right', shadow=True, fontsize='medium')
plt.savefig('plots/escape_latency.svg', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/escape_latency.png', dpi=200, bbox_inches='tight', transparent=True)
