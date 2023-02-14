# basic imports
import gym
import numpy as np
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface


class InterfaceChoice(AbstractInterface):
    
    def __init__(self, modules: dict, observations: np.ndarray, choices: list, with_GUI: bool = True):
        '''
        This is the Open AI gym interface class. The interface wraps the control path and ensures communication between the agent and the environment.
        The class descends from gym.Env, and is designed to be minimalistic (currently!).
        
        Parameters
        ----------
        modules :                           Contains framework modules.
        observations :                      A numpy array containing the observations.
        choices :                           A list of different choice trials.
        with_GUI :                          If true, observations and policy will be visualized.on.
        
        Returns
        ----------
        None
        '''
        super().__init__(modules, with_GUI)
        # observations and choice trials
        self.observations = observations
        self.choices = choices
        # define action space
        self.action_space = gym.spaces.Discrete(2)
        # retrieve observation space
        self.observation_space = list(self.observations.shape[1:])
        self.observation_space[0] *= 2
        self.observation_space = tuple(self.observation_space)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.observation_space)
        # the current observation
        self.observation = np.zeros(self.observation_space.shape)
        # the reward function
        self.rewards = np.zeros(2)
    
    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.
        
        Parameters
        ----------
        action :                            The action selected by the agent.
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        reward :                            The reward received.
        end_trial :                         Flag indicating whether the trial ended.
        logs :                              The (empty) logs dictionary.
        '''
        return np.zeros(self.observation_space.shape), self.rewards[action], True, {}
         
    def reset(self) -> np.ndarray:
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        # randomly select a choice trial
        choice = self.choices[np.random.randint(len(self.choices))]
        # randomly pick order of correct/incorrect observations
        correct_choice = np.random.randint(2)
        # prepare the observations of the current trial
        correct, incorrect = self.observations[choice['correct']], self.observations[choice['incorrect']]
        if correct_choice == 0:
            self.observation = np.concatenate((correct, incorrect))
        else:
            self.observation = np.concatenate((incorrect, correct))
        # update the reward function
        self.rewards.fill(0)
        self.rewards[correct_choice] = 1.
        
        return self.observation
