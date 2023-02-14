# basic imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
import pickle
import os


if __name__ == '__main__':
    # params
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}
    mat.rc('font', **font)
    image_dims = [(16,16,3), (32,32,3), (64,64,3), (128,128,3), (256,256,3)]
    labels = ['16x16', '32x32', '64x64', '128x128', '256x256']
    agents = ['DQN', 'EC']
    
    # make sure that the directory for plotting the results exists
    os.makedirs('plots/', exist_ok=True)
    
    # plot the results
    for agent in agents:
        data_dir = 'data/sps_%s_TMaze.pkl' % agent
        with open(data_dir, 'rb') as data_harbour:
            data = pickle.load(data_harbour)
        # draw
        fig, axs = plt.subplots(figsize=(8.5, 6))
        x = np.arange(1, len(image_dims)+1)
        w = 0.25
        # online
        axs.bar(x, height=1000.0/np.mean(data['online'],axis=0), width=w, color='g', align='center', label='Online Rendering (Unity)')
        # offline
        axs.bar(x+w, height=1000.0/np.mean(data['offline'], axis=0), width=w, color='b', align='center', label='Offline Rendering')
        axs.set_title('%s, TMaze' % agent)
        axs.legend()
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_ylim([0, 50])
        axs.set_xlabel('Input image size')
        axs.set_ylabel('Mean frame time [ms]')
        axs.set_xticks(x+w/2)
        axs.set_xticklabels(labels)
        plt.savefig('plots/sps_%s_TMaze.png' % agent, dpi=200, bbox_inches='tight')
        plt.savefig('plots/sps_%s_TMaze.svg' % agent, dpi=200, bbox_inches='tight')
    plt.show()
