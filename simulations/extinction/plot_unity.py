# basic imports
import os
import pickle
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt


## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
mat.rc('font', **font)

# load CRC and EL
crc_path = 'data/TMaze_extinction_CRC.pkl'
el_path = 'data/TMaze_extinction_EL.pkl'

# ensure that the plots directiory exists
os.makedirs('plots/', exist_ok=True)

# Plot CRC curve
with open(crc_path, 'rb') as handle:
    crc = pickle.load(handle)
crc = np.array(crc)
mu = crc.mean(axis=0)
std = crc.std(axis=0)
fig, axs = plt.subplots(figsize=(8,6))
axs.plot(np.arange(1, len(mu)+1), mu, 'b')
axs.fill_between(np.arange(1, len(mu)+1), mu+std, mu-std, alpha=0.4, color='b')
plt.axvline(100, linestyle='--', color='gray', linewidth=2.5)
axs.set_xlim(1, 200)
axs.set_ylim(0, 150)
axs.set_xlabel('Trial')
axs.set_ylabel('Cumulative response')
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('plots/TMaze_CRC.svg', format='svg')

# Plot escape latency curve
with open(el_path, 'rb') as handle:
    el = pickle.load(handle)
el = np.array(el)
mu = el.mean(axis=0)
std = el.std(axis=0)
fig, axs = plt.subplots(figsize=(8,6))
axs.plot(np.arange(1, len(mu)+1), mu, 'r')
axs.fill_between(np.arange(1, len(mu)+1), mu+std, mu-std, alpha=0.4, color='r')
plt.axvline(100, linestyle='--', color='gray', linewidth=2.5)
axs.set_xlabel('Trial')
axs.set_ylabel('Escape latency')
axs.set_xlim(1, 200)
axs.set_ylim(8, 120)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('plots/TMaze_EL.svg', format='svg')
