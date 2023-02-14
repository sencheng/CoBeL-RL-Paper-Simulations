# basic imports
import os
import time
import pickle
import numpy as np
import pyqtgraph as pg
# tensorflow
import tensorflow as tf
from tensorflow.compat.v1.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Reshape
# framework imports
from cobel.frontends.frontends_unity import FrontendUnityInterface, FrontendUnityOfflineInterface
from cobel.spatial_representations.topology_graphs.four_connected_graph_rotation import FourConnectedGraphRotation
from cobel.agents.em_control import EMControl
from cobel.agents.keras_rl.dqn import DQNAgentBaseline
from cobel.observations.image_observations import ImageObservationUnity
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import ResponseMonitor, EscapeLatencyMonitor

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
backend.set_image_data_format(data_format='channels_last')

# configuration for GPU computing
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

# shall the system provide visual output while performing the experiments?
visual_output = True


def reward_callback(values: dict) -> (float, bool):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    
    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # maximum number of simulation steps
    max_step = 600
    # retrieve the number of steps
    elaped_steps = len(values['episode_traj']) - 1
    
    end_trial = values['current_node'].goal_node
    # force trials to end when the maximum number of steps was reached
    if elaped_steps == max_step:
        end_trial = True

    return values['current_node'].node_reward_bias, end_trial


start_goal_nodes = {'TMaze': [[3], [11]], 'TMaze_LV1': [[5], [15]], 'TMaze_LV2': [[8], [21]],
                    'DoubleTMaze': [[15], [32]], 'TunnelMaze_New': [[44], [101]]}

def single_run(agent: str, env_name: str, image_dims: tuple, offline=False) -> float:
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    agent :                             The type of agent that will be used. Has to be either \'DQN\' or \'EC\'.
    env_name :                          The simulation environment.
    image_dims :                        The dimensions of the imaged that will be rendered.
    offline :                           If true, the simulations will be performed using prerendered images.
    
    Returns
    ----------
    sps :                               The average number of simulation steps per second.
    '''
    # the length of the edge in the topology graph
    step_size = 1.0
    num_trials = 100
    max_steps = 200
    # this is the main window for visual output
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = pg.GraphicsWindow(title="Unity demo" )

    # determine environment
    env = env_name + '%s_%s' % (image_dims[0], image_dims[1])
    
    # a dictionary that contains all employed modules
    modules = dict()
    if not offline:
        modules['world'] = FrontendUnityInterface(env)
    else:
        env_path = os.path.abspath('') + '/../../resources/unity/offline/%s_ss%s_Infos.pickle' % (env, step_size)
        modules['world'] = FrontendUnityOfflineInterface(env_path)
    modules['observation'] = ImageObservationUnity(modules['world'], main_window, image_dims=image_dims, with_GUI=visual_output)
    modules['spatial_representation'] = FourConnectedGraphRotation(modules, {'start_nodes': start_goal_nodes[env_name][0], 'goal_nodes': start_goal_nodes[env_name][1],
                                                                             'start_ori': 90, 'clique_size':4}, step_size=step_size)
    for node in start_goal_nodes[env_name][1]:
        modules['spatial_representation'].nodes[node].goal_node = True
        modules['spatial_representation'].nodes[node].node_reward_bias = 1.
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output, reward_callback)
    
    # initialize monitors
    response_monitor = ResponseMonitor(num_trials, main_window, visual_output)
    el_monitor = EscapeLatencyMonitor(num_trials, max_steps, main_window, visual_output)
    # prepare custom callbacks
    custom_callbacks = {'on_trial_end': [response_monitor.update, el_monitor.update]}

    # initialize RL agent
    if agent == 'DQN':
        # build model (we use the same CNN that was described by Mnih et al. (2015))
        nb_actions = 6
        sensory_model = Sequential()
        sensory_model.add(Reshape(image_dims, input_shape=(1,) + image_dims))
        sensory_model.add(Convolution2D(16, kernel_size=(8, 8), strides=4, activation='relu', padding='same'))
        sensory_model.add(Convolution2D(32, kernel_size=(4, 4), strides=2, activation='relu', padding='same'))
        sensory_model.add(Flatten())  # dimension: 3136
        feature_input = sensory_model.output
        x = Dense(256, activation='relu')(feature_input)
        x = Dense(nb_actions, activation='linear')(x)
        NN_model = Model(inputs=sensory_model.input, outputs=x)
        # initialize DQN
        rl_agent = DQNAgentBaseline(modules['rl_interface'], 5000, 0.1, model=NN_model, custom_callbacks=custom_callbacks)
    elif agent == 'EC':
        rl_agent = EMControl(interface_OAI=modules['rl_interface'], epsilon=0.1, memoryCapacity=20000, custom_callbacks=custom_callbacks)

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent

    # let the agent learn and record the time
    start = time.time()
    rl_agent.train(num_trials)
    end = time.time()
    
    # compute steps per second
    total_steps = np.sum(el_monitor.latency_trace)
    sps = total_steps / (end-start)
    
    backend.clear_session()
    
    modules['world'].stop_unity()
    
    # if visual output is required, activate an output window
    if visual_output:
        main_window.close()

    return sps

if __name__ == '__main__':
    # params
    image_dims = [(16,16,3), (32,32,3), (64,64,3), (128,128,3), (256,256,3)]
    agents = ['EC', 'DQN']
    epochs = 5
    
    # make sure that the directory for storing the results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    for agent in agents:
        sps_record = {'online':[], 'offline':[]}
        for epoch in range(epochs):
            sps_record['online'].append([])
            sps_record['offline'].append([])
            for im_dim in image_dims:
                online_average_sps = single_run(agent, 'TMaze', im_dim, False)
                offline_average_sps = single_run(agent, 'TMaze', im_dim, True)
                sps_record['online'][-1].append(online_average_sps)
                sps_record['offline'][-1].append(offline_average_sps)
        # store simulation results
        pickle.dump(sps_record, open('data/sps_%s_TMaze.pkl' % agent, 'wb'))    
