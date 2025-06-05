from typing import Tuple, Any, List, Dict
from time import time
import numpy as np
from tqdm import tqdm

from src.main.envs.grid_world import GridWorldEnv
from src.main.rl_agents.random_agent import RandomAgent
from src.main.utility.utils import Helpers
from src.main.configs import global_configs
from src.main.utility.chart_results import ChartResults


class QLearningAgent(RandomAgent):
    """
    Q-Learning Agent
    """
    def __init__(
            self,
            seed=100,
            render_mode="human",
            size=5,
            agent_name="Q-learning Agent",
            max_train_episodes=15000,
            max_eval_episodes=10,
            max_train_steps_per_episode=100,
            discount_factor=1.0,
            learning_rate=0.1,
            exploration_rate=1.0,        # Initial exploration rate
            max_exploration_rate=1.0,    # Maximum exploration probability
            min_exploration_rate=0.01,   # Minimum exploration probability
            decay_rate = 0.005, # Exponential decay rate for exploration prob
    ):
        """
        Constructor
        """
        super().__init__(
            seed,
            render_mode,
            size,
            max_train_episodes,
            max_train_steps_per_episode,
            agent_name
        )
        self._max_eval_episodes = max_eval_episodes
        self._max_train_steps_per_episode = max_train_steps_per_episode
        self._discount_factor = discount_factor
        self._learning_rate = learning_rate
        self._epsilon = exploration_rate
        self._max_epsilon = max_exploration_rate
        self._min_epsilon = min_exploration_rate
        self._decay_rate = decay_rate
        self._q_table = self._createQTable()
        self._state_map = self._createStateMap()

    def getAction(
            self,
            observation: Any,
            epsilon: float
    ):
        """
        Computes the action
        :param observation: Observation from the environment
        :param epsilon: Epsilon
        :return: Action
        """
        state = self._observation2State(observation)
        exp_exp_tradeoff = np.random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(self._q_table[state, :])
        # Else doing a random choice --> exploration
        else:
            action = self._env.action_space.sample()
        return action

    def _createStateMap(self) -> Dict[Tuple[int], int]:
        """
        Creates the state map
        :return: State map
        """
        x_t = [(i, j) for i in range(self._grid_size) for j in range(self._grid_size)]
        x_map = {coords: i for i, coords in enumerate(x_t)}
        return x_map

    def _observation2State(self, observation) -> Tuple[int, int]:
        """
        Converts the observation from the environment to the state map
        :param observation: Observation from the environment
        :return: State value
        """
        observation_tuple = (observation[0], observation[1])
        state = self._state_map[observation_tuple]
        return state

    def _createQTable(self) -> np.ndarray:
        """
        Creates Q-Table
        :return: Q-Table
        """
        return np.zeros((self._state_size, self._action_size))

    def updateQTable(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            next_observation: np.ndarray
    ):
        """
        Updates Q-Table:
            - Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            - qtable[new_state,:] : all the actions we can take from new state
        :param observation: Observation from the environment
        :param action: Action
        :param reward: Reward
        :param next_observation: Next Observation from the environment
        """
        state = self._observation2State(observation)
        next_state = self._observation2State(next_observation)
        self._q_table[state, action] = self._q_table[state, action] + self._learning_rate * (
                    reward + self._discount_factor * np.max(self._q_table[next_state, :])
                    - self._q_table[state, action])

    def optimizeEpsilon(self, episode: int) -> float:
        """
        Optimizes Epsilon - reduce epsilon (because we need less and less exploration)
        :param episode: Episode
        :return: New epsilon
        """
        epsilon = self._min_epsilon + (self._max_epsilon - self._min_epsilon) * np.exp(-self._decay_rate * episode)
        return epsilon

    def run(self) -> Tuple[List[float], List[float], List[int]]:
            """
            Executes the agent
            :return: Total reward, average reward and total steps of the agent
            """
            start_time = time()

            for episode in tqdm(range(self._max_episodes), desc="Episode"):
                self._env.action_space.seed(self._seed)
                state, _ = self._env.reset(seed=self._seed)
                observation = state["agent"]
                self._steps = 0
                self._total_reward = 0
                while self._steps < self._max_steps_per_episode:
                    action = self.getAction(observation, self._epsilon)
                    new_obs_as_dict, reward, done, terminated, info = self._env.step(action)
                    new_observation = new_obs_as_dict["agent"]
                    self.updateQTable(observation, action, reward, new_observation)
                    observation = new_observation
                    self._steps += 1
                    self._total_reward += reward
                    # self._env.render()
                    if done:
                        break
                self._average_reward = self._total_reward / self._steps
                self._avg_reward_per_episode.append(self._average_reward)
                self._cum_reward_per_episode.append(self._total_reward)
                self._steps_per_episode.append(self._steps)
                self._epsilon = self.optimizeEpsilon(episode)

            # Helpers.serialObject(self._q_table, global_configs.GRIDWORLD_QL_MODEL_FILE_PATH)
            end_time = time()
            elapsed_time = end_time - start_time
            print(f"Completed the simulation of Grid World, compute time was: {elapsed_time:.2f} seconds")
            return self._avg_reward_per_episode, self._cum_reward_per_episode, self._steps_per_episode

    @property
    def epsilon(self) -> float:
        """
        Getter for epsilon
        :return: epsilon
        """
        return self._epsilon

    @property
    def max_eval_episodes(self) -> int:
        """
        Getter for max evaluation number of episodes
        :return: max evaluation number of episodes
        """
        return self._max_episodes

    @property
    def max_train_steps_per_episode(self) -> int:
        """
        Getter for max training steps per episode
        :return: max training steps per episode
        """
        return self._max_train_steps_per_episode

    @property
    def env(self) -> GridWorldEnv:
        """
        Getter for current environment
        :return: current environment
        """
        return self._env

    @property
    def seed(self) -> int:
        """
        Getter for random seed
        :return: random seed
        """
        return self._seed

    @property
    def q_table(self) -> np.ndarray:
        """
        Getter for current Q table
        :return: current Q table
        """
        return self._q_table

    @q_table.setter
    def q_table(self, value: np.ndarray):
        """
        Setter for current Q table
        :param value: q_table
        """
        self._q_table = value

