# basic imports
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle
import os
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# CoBel-RL framework
from cobel.agents.dyna_dqn import DynaDSR
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor, RepresentationMonitor
from cobel.misc.gridworld_tools import make_gridworld


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
    model.add(Dense(units=64, input_shape=input_shape, activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dense(units=output_units, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

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
    logs['rl_parent'].interface_OAI.world['rewards'].fill(0.)
    if logs['trial'] >= 100:
        logs['rl_parent'].interface_OAI.world['rewards'][15] = 10.

def prepare_invalid_transitions() -> list:
    '''
    This functions prepares the invalid transitions for the Blogget maze.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    invalid_transitions :               A list containing invalid state-transition tuples.
    '''
    # build maze walls
    invalid_transitions = [(32, 33),(32, 24),(33, 34),(34, 26),(35, 27),(36, 28),(36, 37),(37, 38),
                           (24, 16),(26, 18),(28, 20),(29, 30),(25, 26),(16, 8),(16, 17),(17, 18),
                           (18, 10),(18, 19),(19, 20),(20, 12),(20, 21),(21, 22),(22, 14),(23, 15),
                           (8, 0),(9, 1),(10, 2),(12, 4),(13, 5),(14, 6),(15, 7),(11, 12),(3, 2),(3, 4)]
    # reverse transitions
    rev = []
    for transition in invalid_transitions:
        rev.append((transition[1], transition[0]))
    invalid_transitions += rev
    # one-way transitions
    invalid_transitions += [(25, 24), (9, 17), (11, 10), (27, 19), (29, 28), (13, 21)]
    
    return invalid_transitions
    

def learning_paradigm(type_of_learning: str = 'latent') -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    type_of_learning :                  The learning paradigm that will be used. \'latent\' by default.
    
    Returns
    ----------
    latency_trace :                     The escape latency trace.
    activtiy_trace_left :               The activity trace for the left action (action == 0).
    activtiy_trace_up :                 The activity trace for the up action (action == 1).
    activtiy_trace_right :              The activity trace for the right action (action == 2).
    activtiy_trace_down :               The activity trace for the down action (action == 3).
    '''
    np.random.seed()    
    # learning paradigm specific settings
    if type_of_learning == 'latent':
        # amount of trials
        number_of_trials = 200
        # reward at goal state
        reward = np.array([[15, 0.0]])
    elif type_of_learning == 'direct':
        number_of_trials = 100
        reward = np.array([[15, 10.0]])

    # Blogget latent learning
    invalid_transitions = prepare_invalid_transitions()  

    world = make_gridworld(5, 8, terminals=[15], starting_states=np.array([24]), invalid_transitions=invalid_transitions, rewards=reward, goals=[15])
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # maximum steps per trial
    max_steps = 200
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, None, False, [0, 10])
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, None, False)
    # initialize representation monitors
    representation_monitors = {}
    for action in range(4):
        representation_monitors[action] = RepresentationMonitor(observations={0: np.eye(40)} , dimensions=(world['height'], world['width']), model=None, layer=-1, units=np.arange(40).astype(int), update_interval=100, gui_parent=None, visual_output=False)
        representation_monitors[action].last_update = -1

    # build models
    model_SR = SequentialKerasNetwork(build_model((40,), 40))
    model_reward = SequentialKerasNetwork(build_model((40,), 1))
    
    # define callbacks
    custom_callbacks = {'on_trial_end': [reward_monitor.update, escape_latency_monitor.update] + [representation_monitors[action].update for action in range(4)]}
    if type_of_learning == 'latent':
        custom_callbacks['on_trial_end'].append(trial_end_callback)

    # initialize RL agent
    rl_agent = DynaDSR(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5, gamma=0.9,
                       model_SR=model_SR, model_reward=model_reward, custom_callbacks=custom_callbacks)
    
    rl_agent.mask_actions = True
    rl_agent.ignore_terminality = True
    # rl_agent.policy = 'softmax'
    # rl_agent.use_Deep_DR = True
    
    # set representation monitor models
    for action in range(4):
        representation_monitors[action].model = rl_agent.models_online[action]

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=100)
    
    # clear keras session (for performance)
    K.clear_session()

    return escape_latency_monitor.latency_trace, representation_monitors[0].activity_trace, representation_monitors[1].activity_trace, representation_monitors[2].activity_trace, representation_monitors[3].activity_trace


if __name__ == '__main__':
    # params
    number_of_runs = 20
    paradigms = ['latent', 'direct']
    variables = ['escape_latency', 'activity_left', 'activity_up', 'activity_right', 'activity_down']
    
    # ensure that the directory for storing the results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    for paradigm in paradigms:
        results = {var: [] for var in variables}
        print('Running simulations in \'%s\' paradigm.' % paradigm)
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\tRun: %d' % (run + 1))
            simulation_data = learning_paradigm(paradigm)
            for var, data in zip(variables, simulation_data):
                results[var].append(data)
        # store results
        pickle.dump(results, open('data/results_%s.pkl' % paradigm, 'wb')) 
