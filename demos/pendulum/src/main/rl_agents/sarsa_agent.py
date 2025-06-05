from typing import Tuple, Any, List, Dict
from time import time
import numpy as np
from tqdm import tqdm

from src.main.utility.utils import Helpers
from src.main.configs import global_configs as configs
from src.main.utility.chart_results import ChartResults
from src.main.utility.logging import Logger
from src.main.rl_agents.q_learning_agent import QLearningAgent

class SarsaAgent(QLearningAgent):
    """
    Sarsa RL agent
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
        :param n_episodes: Number of episodes
        :param max_steps: Maximum number of steps
        """
        super().__init__(
            env=env,
            seed=seed,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            n_episodes=n_episodes,
            max_steps=max_steps
        )

    def update(self, s, a, r, s_next,a_next):
        """
        Update the Q-value
        :param s: State
        :param a: Action
        :param r: Reward
        :param s_next: Next state
        :param a_next: Next action
        :return:
        """
        td_target = r + self.gamma * self.Q[s_next, a_next]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def train(self):
        """
        Train the Q-learning agent
        :return:
        """
        rewards = []
        for ep in tqdm(range(self.n_episodes), desc="Training episodes:"):
            s, _ = self.env.reset()  # start new episode
            a = self.chooseAction(s)
            total_reward = 0

            for _ in range(self.max_steps):
                s_next, r, done, _, _ = self.env.step(a)
                a_next = self.chooseAction(s_next)
                self.update(s, a, r, s_next, a_next)
                s, a = s_next, a_next
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
