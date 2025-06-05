
from typing import Tuple, List
from time import time
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.main.envs.grid_world import GridWorldEnv

class RandomAgent:
    """
    RandomAgent specification
    """
    def __init__(
            self,
            seed: int = 100,
            render_mode="human",
            size=5,
            max_episodes=15000,
            max_steps_per_episode=100,
            agent_name="Random Agent",
    ):
        """
        Constuctor
        :param render_mode: Render mode
        :param size: Grid size
        :param max_episodes: Max number of episodes
        :param max_steps_per_episode: Max number of steps per episode
        """
        print(f"Constructing the '{agent_name}' Grid World Agent..")
        self._seed = seed
        self._render_mode = render_mode
        self._grid_size = size
        self._rewards = []
        self._steps = 0
        self._total_reward = 0
        self._average_reward = 0
        self._obs = None
        self._env = GridWorldEnv(render_mode=render_mode, size=size)
        self._action_size = int(self._env.action_space.n)
        self._state_size = self._grid_size**2
        print(f"action_size: {self._action_size}\tstate_size: {self._state_size}")
        self._max_episodes = max_episodes
        self._max_steps_per_episode = max_steps_per_episode

        self._avg_reward_per_episode = []
        self._cum_reward_per_episode = []
        self._steps_per_episode = []

    def run(self) -> Tuple[List[float], List[float], List[int]]:
        """
        Executes the agent
        :return: Total reward, average reward and total steps of the agent
        """
        start_time = time()

        for episode in tqdm(range(self._max_episodes),desc="Episode"):
            self._obs, _ = self._env.reset(self._seed)
            self._steps = 0
            self._total_reward = 0
            while self._steps < self._max_steps_per_episode:
                action = self._env.action_space.sample()
                self._obs, reward, done, terminated, info = self._env.step(action)
                self._steps += 1
                self._total_reward += reward
                self._env.render()
                if done:
                    break
            self._average_reward = self._total_reward / self._steps
            self._avg_reward_per_episode.append(self._average_reward)
            self._cum_reward_per_episode.append(self._total_reward)
            self._steps_per_episode.append(self._steps)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"Completed the simulation of Grid World, compute time was: {elapsed_time:.2f} seconds")
        return self._avg_reward_per_episode, self._cum_reward_per_episode, self._steps_per_episode

    def plotRewardCurves(self):
        """
        Plot reward curves
        """
        plt.plot(self._avg_reward_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.show()

