# basic imports
import os
import numpy as np
import pyqtgraph as qg
import pandas as pd
import pickle
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Reshape, Activation, Dropout
# framework imports
from cobel.agents.keras_rl.dqn import DQNAgentBaseline
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
# custom modules
from custom_modules import BlenderOnlineRenderer
from custom_modules import ImageObservationFOV
from custom_modules import HexagonalGraphAllocentric


# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False

df = pd.DataFrame(columns=['run', 'trial', 'steps', 'reward']) 
data = []
activations = []   
record_activation_interval = 800
input_file = 'input_images_guidance.npy'
input_img = np.load(input_file)

def build_model(input_shape: tuple, output_units: int) -> Sequential:
    '''
    This function builds a simple network model. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    output_units :                      The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    model = Sequential()
    model.add(Reshape(input_shape, input_shape=(1,) + input_shape, name='reshape'))
    model.add(Convolution2D(32, (5,5), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', name='analysis'))
    model.add(Dropout(0.35))
    model.add(Dense(output_units))
    
    return model

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
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = -1.0
    end_trial = values['currentNode'].goal_node
    if end_trial:
        reward = 1.0
    
    return reward, end_trial

def record_training_data(logs: dict):
    data.append([logs['trial'], logs['steps'], logs['trial_reward']])
    
def record_activations(logs: dict):
    if (logs['trial'] % record_activation_interval) == 0:
        print('\t\tTrial: ', logs['trial'] + 1)
        rl_agent = logs['rl_parent']
        dense_layer_activations = K.function([rl_agent.model.layers[0].input], [rl_agent.model.layers[8].output])
        activation = []
        for angle in input_img:
            activation.append(np.array(dense_layer_activations([angle])))
        activations.append(activation)
    
def single_run(run: int = 1):
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    print("Run : ", run) 
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Guidance')
        
    # determine demo scene path
    demo_scene = '../../resources/blender/guidance_office.blend'
    print(demo_scene)
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = BlenderOnlineRenderer(demo_scene)
    modules['observation'] = ImageObservationFOV(modules['world'], main_window, visual_output,
                                                 image_dims=(72, 12), view_angle=240.0)
    modules['spatial_representation'] = HexagonalGraphAllocentric(n_nodes_x=9, n_nodes_y=9,
                                                       n_neighbors=6, goal_nodes=[38],
                                                       visual_output=True, 
                                                       world_module=modules['world'],
                                                       use_world_limits=True, 
                                                       observation_module=modules['observation'], 
                                                       rotation=True)
    modules['spatial_representation'].set_visual_debugging(main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output, reward_callback)
    
    # amount of trials
    number_of_trials = 6000
    # maximum steps per trial
    max_steps = 100
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-max_steps, 10])
    
    # build model
    #model = build_model(modules['rl_interface'].observation_space.shape, modules['rl_interface'].action_space.n)
    model = build_model((12, 48, 3), modules['rl_interface'].action_space.n)
    
    # prepare custom callbacks
    custom_callbacks = {'on_trial_end': [reward_monitor.update, record_training_data, record_activations]}
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 3000, 0.3, model=model, custom_callbacks=custom_callbacks)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    rl_agent.train(number_of_trials, max_steps)
    
    # stop simulation
    modules['world'].stop_blender()
    
    # prepare simulation data for saving
    x = np.array(data)
    x = np.hstack((int(run)*np.ones(number_of_trials).reshape(number_of_trials,1), x))
    logs = df.append(pd.DataFrame(x, columns=df.columns), ignore_index=True)
    logs.to_csv('logs.csv', sep='\t', encoding='utf-8',index=False,
              mode='a', header=not os.path.exists('logs.csv'))
    # save simulation data
    with open('data/activations/activations_{}.pkl'.format(run), 'wb') as f:
        pickle.dump(activations, f)
        
    # clear keras session (for performance)
    K.clear_session()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    # make sure that the directories for storing the activity exists
    os.makedirs('data/activations', exist_ok=True)
    
    # run simulations
    print('Running simulations.')
    for run in range(10):
        print('Run: ', run + 1)
        single_run(run)
