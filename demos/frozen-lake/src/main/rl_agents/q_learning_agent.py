from typing import Tuple, Any, List, Dict
from time import time
import numpy as np
from tqdm import tqdm
import random

from src.main.utility.utils import Helpers
from src.main.configs import global_configs as configs
from src.main.utility.chart_results import ChartResults
from src.main.utility.logging import Logger

np.random.seed(100)
random.seed(100)

class QLearningAgent:
    """
    Q-Learning RL agent
    """
    def __init__(
        self,
        env,
        seed: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay:float,
        min_epsilon:float,
        n_episodes: int,
        max_steps: int
        ):
        """
        Constructor.
        :param env: Environment
        :param seed: Random seed
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Epsilon
        :param epsilon_decay: Epsilon decay factor
        :param min_epsilon: Minimum epsilon
        """
        self.logger = Logger.getLogger()
        self.env = env
        self.seed = seed
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.Q = np.zeros((n_states, n_actions)) # + np.random.rand(n_states, n_actions)

        self.logger.info(f"Frozen Lake environment creation..")
        self.logger.info(f"Observation space: {n_states}")
        self.logger.info(f"Action space: {n_actions}")
        self.logger.info(f"""RL hyperparameters are:
                  \nalpha: {self.alpha}
                  \nepsilon: {self.epsilon}
                  \nepsilon_decay: {self.epsilon_decay}
                  \nmin_epsilon: {self.min_epsilon}\n""")

    def chooseAction(self, state):
        """
        Choose the RL action
        :param state: Input state
        :return: Action
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next):
        """
        Update the Q-value
        :param s: State
        :param a: Action
        :param r: Reward
        :param s_next: Next state
        :return:
        """
        td_target = r + self.gamma * np.max(self.Q[s_next, :])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def computeDecayEpsilon(self):
        """
        computes the decay epsilon
        :return: Epsilon
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self):
        """
        Train the Q-learning agent
        :return:
        """
        rewards = []
        for ep in tqdm(range(self.n_episodes), desc="Training episodes:"):
            s, _ = self.env.reset()  # start new episode

            total_reward = 0

            for _ in range(self.max_steps):
                a = self.chooseAction(s)
                s_next, r, done, _, _ = self.env.step(a)
                self.update(s, a, r, s_next)
                s = s_next

                total_reward += r
                if done:
                    break

            self.computeDecayEpsilon()
            rewards.append(total_reward)
            if (ep + 1) % configs.EPISODE_UPDATE_FREQUENCY == 0:
                print(f"Episode {ep + 1}/{self.n_episodes}  Average Reward: "
                      f"{np.mean(rewards[-configs.EPISODE_UPDATE_FREQUENCY:]):.3f}")
                print(f"\nQ: {self.Q}\n")
        return rewards
