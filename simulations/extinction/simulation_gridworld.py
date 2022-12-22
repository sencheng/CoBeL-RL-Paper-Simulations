# basic imports
import numpy as np
import pickle
import os
import pyqtgraph as qg
# framework imports
from cobel.agents.dyna_q import DynaQAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import EscapeLatencyMonitor, ResponseMonitor, TrajectoryMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
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
    logs['response'] = int(logs['rl_parent'].interface_OAI.current_state == 6)
    if logs['trial'] == 100:
        logs['rl_parent'].interface_OAI.world['rewards'].fill(0.)
        logs['rl_parent'].interface_OAI.world['rewards'][0] = 1.

def single_run() -> (np.ndarray, np.ndarray, list):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    escape_latency :                    The escape latency trace.
    responses :                         The response trace.
    trajectory :                        The trajectory trace.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Changing Goal Position')
        
    # load gridworld T-maze
    world = pickle.load(open('t_maze.pkl', 'rb'))
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = 200
    # maximum steps per trial
    max_steps = 35
    
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    response_monitor = ResponseMonitor(number_of_trials, main_window, visual_output)
    trajectory_monitor = TrajectoryMonitor(max_steps, main_window, visual_output)
    
    # define callbacks
    custom_callbacks = {'on_trial_end': [trial_end_callback, escape_latency_monitor.update, response_monitor.update], 'on_step_end': [trajectory_monitor.update]}
    
    # initialize RL agent
    rl_agent = DynaQAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5,
                          learning_rate=0.9, gamma=0.9, custom_callbacks=custom_callbacks)

    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=5)
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    return escape_latency_monitor.latency_trace, response_monitor.responses, trajectory_monitor.trajectory_trace

if __name__ == '__main__':
    # params
    number_of_runs = 100
    
    # run simulations in T-maze environment
    print('Running simulations in gridworld T-maze.')
    results = {'escape_latency': [], 'responses': [], 'trajectory': []}
    for run in range(number_of_runs):
        if (run + 1) % 10 == 0:
            print('\tRun: ' + str(run + 1))
        latency, response, trajectory = single_run()
        results['escape_latency'].append(latency)
        results['responses'].append(response)
        results['trajectory'].append(trajectory)
    os.makedirs('data/', exist_ok=True)
    pickle.dump(results, open('data/results_gridworld.pkl', 'wb'))
