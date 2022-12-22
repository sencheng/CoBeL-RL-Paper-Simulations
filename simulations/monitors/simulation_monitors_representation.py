# basic imports
import numpy as np
import pickle
import os
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# CoBel-RL framework
from cobel.agents.dqn import SimpleDQN
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from cobel.interfaces.discrete import InterfaceDiscrete
from cobel.analysis.rl_monitoring.rl_performance_monitors import RepresentationMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def build_model(input_shape, number_of_actions):
    '''
    This function builds a simple network model. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    number_of_actions :                 The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    model = Sequential()
    model.add(Dense(units=64, input_shape=input_shape, activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dense(units=number_of_actions, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: Representation Monitor')
    
    # load gridworld T-maze
    world = pickle.load(open('t_maze.pkl', 'rb'))
    
    # we use a one-hot encoding for the states
    observations = np.eye(world['states'])
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = InterfaceDiscrete(modules, world['sas'], observations, world['rewards'], world['terminals'],
           world['starting_states'], world['coordinates'], world['goals'], visual_output, main_window)
    
    # amount of trials
    number_of_trials = 200
    # maximum steos per trial
    max_steps = 25
    
    # initialize monitors
    response_monitor = RepresentationMonitor({0: observations}, (9, 9), units=np.arange(64).astype(int), gui_parent=main_window, visual_output=visual_output)
    
    # adjust visualization
    main_window.setGeometry(50, 50, 1800, 600)
    
    # define custom callbacks
    custom_callbacks = {'on_trial_end': [response_monitor.update]}
    
    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], epsilon=0.3, beta=5, gamma=0.9,
                         model=SequentialKerasNetwork(build_model((81,), 4)), custom_callbacks=custom_callbacks)
    response_monitor.model = rl_agent.model_online
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop visualization
    if visual_output:
        main_window.close()
        
    return response_monitor.activity_trace


if __name__ == '__main__':
    activity = single_run()
    os.makedirs('data/', exist_ok=True)
    pickle.dump(activity, open('data/activity.pkl', 'wb'))
