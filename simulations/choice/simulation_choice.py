# basic imports
import numpy as np
import pickle
import os
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# framework imports
from cobel.agents.keras_rl.dqn import DQNAgentBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
# local imports
from interface_choice import InterfaceChoice


def single_run(observations: np.ndarray, choice: list) -> (np.ndarray, np.ndarray, list):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    observations :                      A numpy array containing the observations.
    choices :                           A list of different choice trials.
    
    Returns
    ----------
    rewards :                           The reward trace.
    '''
    np.random.seed()   
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceChoice(modules, observations, choices, False)
    
    # amount of trials
    number_of_trials = 50
    # maximum steps per trial
    max_steps = 35
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, None, False, [0, 1])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 1000000, 0.3, None, custom_callbacks={'on_trial_end': [reward_monitor.update]})
    rl_agent.agent.nb_steps_warmup = 1
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
        
    return reward_monitor.reward_trace

if __name__ == '__main__':
    # params
    number_of_runs = 100
    observations = np.eye(4)
    choices = [{'correct': 0, 'incorrect': 1}, {'correct': 2, 'incorrect': 3}]
    
    # run simulations in T-maze environment
    print('Running simulations.')
    rewards = []
    for run in range(number_of_runs):
        if (run + 1) % 10 == 0:
            print('\tRun: ' + str(run + 1))
        rewards.append(single_run(observations, choices))
    os.makedirs('data/', exist_ok=True)
    pickle.dump(np.array(rewards), open('data/rewards.pkl', 'wb'))
