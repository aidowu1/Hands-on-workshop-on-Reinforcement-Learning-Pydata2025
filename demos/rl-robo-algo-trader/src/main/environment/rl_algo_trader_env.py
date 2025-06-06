from typing import List, Tuple, Any, Dict
import math
import gymnasium as gym
import random
import numpy as np
import pandas as pd

from src.main.utility.utils import Helpers
import src.main.configs.global_configs as configs


class TradingEnv(gym.Env):
    """
    RL asset trading environment
    """

    def __init__(
            self,
            symbol: str,
            features: List[str],
            window: int,
            lags: int,
            leverage: int =1,
            min_performance: float =0.5,
            start: int =0,
            end: int =None,
            mu: float =None,
            std: float =None):
        """
        Constructor
        :param symbol: Asset symbol
        :param features: Features
        :param window: Data window
        :param lag: Lag
        :param leverage: Leverage
        :param min_performance: Minimum performance
        :param start: start
        :param end: End
        :param mu: Mean
        :param std: Standard deviation
        """
        self.symbol = symbol
        self.features = features
        self.n_features = len(features)
        self.window = window
        self.lags = lags
        self.leverage = leverage
        self.min_performance = min_performance
        self.start = start
        self.end = end
        self.mu = mu
        self.std = std
        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(self.lags, self.n_features), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.states_buy = None
        self.states_sell = None
        self.raw = Helpers.getAssetData()
        self._prepare_data()
        self.total_reward = None
        self.episode_infos = []
        self.inventory = None
        self.total_profit = None

    def step(self, action):
        """
        Step function
        :param action: Action taken by the agent
        :return: Observations, reward, done, truncated and infos
        """
        reward = 0
        current_price = self.data[self.symbol].iloc[self.bar]
        if action == 1:  # buy
            self.inventory.append(current_price)
            self.states_buy.append(self.bar)

        elif action == 2 and len(self.inventory) > 0:  # sell
            bought_price = self.inventory.pop(0)
            reward = max(current_price - bought_price, 0)
            self.total_profit += current_price - bought_price
            self.states_sell.append(self.bar)
        self.bar += 1

        state = self.getState()
        info = self._getInfos()
        self.episode_infos.append(info)

        if self.bar >= len(self.data):
            done = True
            self.total_profit = 0
        else:
            done = False
        terminated = False

        return state.values, reward, done, terminated, info


    def reset(self, seed=configs.SEED, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the RL environment
        :param seed: Seed
        :param options: Options
        """
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.states_buy = []
        self.states_sell = []
        self.bar = self.lags
        self.total_reward = 0
        self.inventory = []
        self.total_profit = 0
        state = self.data_[self.features].iloc[self.bar-
                                               self.lags:self.bar]
        return state.values, {}

    def _prepare_data(self):
        """
        Prepares the asset data
        """
        self.data = pd.DataFrame(self.raw[self.symbol])
        self.data = self.data.iloc[self.start:]
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        self.data['s'] = self.data[self.symbol].rolling(
            self.window).mean()
        self.data['m'] = self.data['r'].rolling(self.window).mean()
        self.data['v'] = self.data['r'].rolling(self.window).std()
        self.data.dropna(inplace=True)
        if self.mu is None:
            self.mu = self.data.mean()
            self.std = self.data.std()
        self.data_ = (self.data - self.mu) / self.std
        self.data_['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data_['d'] = self.data_['d'].astype(int)
        if self.end is not None:
            self.data = self.data.iloc[:self.end - self.start]
            self.data_ = self.data_.iloc[:self.end - self.start]

    def seed(self, seed):
        """
        Seed to the random number generation
        :param seed: Seed
        """
        random.seed(seed)
        np.random.seed(seed)

    def getState(self) -> np.ndarray:
        """
        Gets the RL state
        :return: State
        """
        return self.data_[self.features].iloc[self.bar -
                                              self.lags:self.bar]

    def _getInfos(self) -> Dict[str, Any]:
        """
        Gets the RL infos
        :return: RL infos
        """
        info = {
            "states_buy": self.states_buy,
            "states_sell": self.states_sell,
            "accuracy": self.accuracy,
            "bar": self.bar,
            "performance": self.performance,
            "profit": self.total_profit,

        }
        return info

    def render(self, mode='human'):
        """
        Renders the environment
        :param mode: Mode
        """
        pass  # Optional visualization

    def close(self):
        """
        Closes the environment
        """
        super().close()

