from typing import List
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np

from src.main.rl_agents.q_learning_agent import QLearningAgent
from src.main.configs.q_learning_agent_configs import QLearningAgentConfig
from src.main.envs.grid_world import GridWorldEnv

class EvaluateRLAgent:
    """
    Class used to evaluate a RL agent.
    """
    def __init__(
            self,
            render_mode = "human",
            agent: QLearningAgent = None,
            max_eval_episodes: int = 3
    ):
        """
        Constructor.
        :param env: Environment
        :param render_mode: render mode
        """
        self._agent = agent
        self._max_eval_episodes = max_eval_episodes
        self._env = self._agent.env
        self._env.render_mode = render_mode
        self._global_rewards = []


    def evaluate(self):
        """
        Evaluates the agent and returns the average reward.
        :return: average reward
        """

        # Q-learning algorithm
        for episode in range(self._max_eval_episodes):
            self._agent.env.action_space.seed(self._agent.seed)

            # Reset the environment between episodes
            state, info = self._agent.env.reset()
            observation = state["agent"]

            done = False  # Keep track if the current episode has finished
            current_rewards = 0  # Keep track of the rewards of the current episode

            # Do for every timestep
            for step in range(self._agent.max_train_steps_per_episode):

                frame = self._agent.env.render()

                # Get the next action from the agent given the current state
                action = self._agent.getAction(observation, self._agent.epsilon)

                # Take the action and get the new state, and information
                new_obs_as_dict, reward, done, terminated, info = self._agent.env.step(action)
                new_observation = new_obs_as_dict["agent"]

                # Set state to the new state and accumulate rewards of this episode (here an episode can have a maximum
                # reward of 1)
                observation = new_observation
                current_rewards += reward

                # If the environment terminates the episode stop
                if done or terminated:
                    title = "The agent fell through a pit!!"
                    frame = self._agent.env.render()
                    if reward == 10:
                        title = "The goal has been reached!!"
                    print(f"{title}")
                    break

        self._agent.env.close()



