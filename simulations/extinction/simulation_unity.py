# basic imports
import numpy as np
import os
import pyqtgraph as pg
import pickle
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Reshape
from tensorflow.keras.optimizers import Adam
# framework imports
from cobel.frontends.frontends_unity import FrontendUnityOfflineInterface
from cobel.spatial_representations.topology_graphs.four_connected_graph_rotation import FourConnectedGraphRotation
from cobel.agents.keras_rl.dqn import DQNAgentBaseline
from cobel.observations.image_observations import ImageObservationUnity
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import ResponseMonitor, EscapeLatencyMonitor

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
K.set_image_data_format(data_format='channels_last')
# shall the system provide visual output while performing the experiments? NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visual_output = False


def trial_end_callback(logs: dict):
    '''
    This is a callback routine that is called when a single trial ends.
    Here, functionality for performance evaluation can be introduced.
    
    Parameters
    ----------
    logs :                              Output of the reinforcement learning subsystem.
    
    Returns
    ----------
    None
    '''
    logs['response'] = int(logs['rl_parent'].interface_OAI.modules['spatial_representation'].current_node == 11)
    if logs['trial'] == 100:
        # change terminals
        logs['rl_parent'].interface_OAI.modules['spatial_representation'].nodes[0].goal_node = True
        logs['rl_parent'].interface_OAI.modules['spatial_representation'].nodes[11].goal_node = False
        # change rewards
        logs['rl_parent'].interface_OAI.modules['spatial_representation'].nodes[0].node_reward_bias = 1.
        logs['rl_parent'].interface_OAI.modules['spatial_representation'].nodes[11].node_reward_bias = 0.

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
    return values['current_node'].node_reward_bias, values['current_node'].goal_node


available_mazes = ['TMaze', 'TMaze_LV1', 'TMaze_LV2', 'DoubleTMaze', 'TunnelMaze_new']
start_goal_nodes = {'TMaze': [[3], [0, 11]], 'TMaze_LV1': [[5], [15]], 'TMaze_LV2': [[8], [21]],
                    'DoubleTMaze': [[15], [32]], 'TunnelMaze_New': [[44], [101]]}

def single_run(running_env: str, num_trials: int, max_steps: int) -> (np.ndarray, np.ndarray):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    running_env :                       The simulation environment.
    num_trials :                        The number of trials that the agent will be trained.
    max_steps :                         The mnaximum number of steps per trial.
    
    Returns
    ----------
    escape_latency :                    The escape latency trace.
    responses :                         The response trace.
    '''
    # the length of the edge in the topology graph
    step_size = 1.0
    observation_space = (84, 84, 3)
    # this is the main window for visual output
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = pg.GraphicsWindow(title='Unity Demo')
    
    # determine world info file path
    worldInfo = os.path.abspath('') + '/../../resources/unity/offline/%s_ss1.0_Infos.pickle' % running_env
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendUnityOfflineInterface(worldInfo)
    modules['observation'] = ImageObservationUnity(modules['world'], main_window, visual_output, False, observation_space)
    modules['spatial_representation'] = FourConnectedGraphRotation(modules, {'start_nodes':start_goal_nodes[running_env][0], 'goal_nodes':start_goal_nodes[running_env][1], 'start_ori': 90, 'clique_size':4}, step_size=step_size)
    modules['spatial_representation'].nodes[0].goal_node = False
    modules['spatial_representation'].nodes[11].goal_node = True
    modules['spatial_representation'].nodes[11].node_reward_bias = 1.
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output, reward_callback)

    # initialize monitors
    response_monitor = ResponseMonitor(num_trials, main_window, visual_output)
    el_monitor = EscapeLatencyMonitor(num_trials, max_steps, main_window, visual_output)
    
    # build model (we use the same CNN that was described by Mnih et al. (2015))
    nb_actions = 6
    sensory_model = Sequential()
    sensory_model.add(Reshape(observation_space, input_shape=(1,) + observation_space))
    sensory_model.add(Convolution2D(16, kernel_size=(8, 8), strides=4, activation='relu'))
    sensory_model.add(Convolution2D(32, kernel_size=(4, 4), strides=2, activation='relu'))
    sensory_model.add(Flatten())  # dimension: 3136
    feature_input = sensory_model.output
    x = Dense(256, activation='relu')(feature_input)
    x = Dense(nb_actions, activation='linear')(x)
    NN_model = Model(inputs=sensory_model.input, outputs=x)
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 10000, 0.1, model=NN_model,
                                custom_callbacks={'on_trial_end': [trial_end_callback, response_monitor.update, el_monitor.update]})

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    # let the agent learn for 100 episode
    rl_agent.train(num_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    # stop Unity
    modules['world'].stop_unity()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

    return response_monitor.CRC, el_monitor.latency_trace

if __name__ == '__main__':
    # params
    num_epochs = 50
    
    # run simulations
    CRC, EL = [], []
    for epoch in range(num_epochs):
        crc, el = single_run('TMaze', 200, 100)
        CRC.append(crc)
        EL.append(el)
    
    # store CRC and EL
    os.makedirs('data/', exist_ok=True)
    pickle.dump(CRC, open('data/TMaze_extinction_CRC.pkl', 'wb'))
    pickle.dump(EL, open('data/TMaze_extinction_EL.pkl', 'wb'))
