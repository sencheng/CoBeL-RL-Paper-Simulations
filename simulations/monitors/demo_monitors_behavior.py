# basic imports
import numpy as np
import pickle
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
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor, ResponseMonitor, TrajectoryMonitor

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

def response_callback(logs: dict):
    '''
    This function determines the response that was emitted by the agent. 
    
    Parameters
    ----------
    logs :                              The trial log dictionary.
    
    Returns
    ----------
    None
    '''
    # check response
    logs['response'] = 1 if logs['rl_agent'].interface_OAI.currentState == 16 else 0

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
        main_window = qg.GraphicsWindow(title='Demo: Behavioral Monitors')
    
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
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 1])
    el_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    response_monitor = ResponseMonitor(number_of_trials, main_window, visual_output)
    trajectory_monitor = TrajectoryMonitor(max_steps, main_window, visual_output)
    
    # adjust visualization
    main_window.setGeometry(50, 50, 1800, 360)
    
    # define custom callbacks
    custom_callbacks = {'on_trial_end': [reward_monitor.update, el_monitor.update, response_monitor.update],
                        'on_step_end': [trajectory_monitor.update]}
    
    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], epsilon=0.3, beta=5, gamma=0.9,
                         model=SequentialKerasNetwork(build_model((81,), 4)), custom_callbacks=custom_callbacks)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop visualization
    if visual_output:
        main_window.close()
        
    return reward_monitor.reward_trace, el_monitor.latency_trace, response_monitor.responses, trajectory_monitor.trajectory_trace


if __name__ == '__main__':
    single_run()
