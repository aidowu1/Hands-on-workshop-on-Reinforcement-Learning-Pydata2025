from typing import List, Tuple, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

import src.main.configs.global_configs as configs
import src.main.configs.rl_robo_trader_run_configs as rl_configs
from src.main.utility.utils import Helpers
from src.main.environment.rl_algo_trader_env import TradingEnv
from src.main.utility.enum_types import RLAgorithmType
from src.main.rl_algorithms.train_evaluate_test.dqn_algorithm import DQNTrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.ppo_algorithm import PPOTrainAlgorithm
from src.main.utility.logging import Logger

logger = Logger.getLogger()

class RLAlgoTradingAgent:
    """
    RL algo trading agent
    """
    def __init__(
            self,
            rl_algorithm_type: RLAgorithmType = RLAgorithmType.dqn
    ):
        """
        Constructor
        """
        self.train_env, self.test_env = self._createAlgoTradingEnvironments()
        self.rl_algorithm_type = rl_algorithm_type
        self.rl_problem_title=configs.RL_ALGO_TRADER_ENV_NAME
        if self.rl_algorithm_type == RLAgorithmType.dqn:
            self.agent = DQNTrainAlgorithm(
                env=self.train_env,
                rl_problem_title=self.rl_problem_title)
        else:
            self.agent = PPOTrainAlgorithm(
                env=self.train_env,
            rl_problem_title=self.rl_problem_title)

    def train(self) -> List[Dict[str, Any]]:
        """
        Trains the RL algo trading agent
        :return: RL infos
        """
        logger.info(f"Training RL algo trading agent with the {self.rl_algorithm_type} algorithm")
        self.agent.train()
        return self.train_env.episode_infos

    def validate(self) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Tests/validate the RL algo trading agent
        :return: RL infos
        """
        rewards, infos = self.agent.evaluate(env=self.test_env)
        info = infos[-1]
        self.reportAgentBehaviour(self.test_env, info)
        return rewards, infos

    def plotTrainRewardCurve(self):
        """
        Plots the train reward curve
        """
        average_rewards = self.agent.smoothed_average_rewards
        plt.plot(average_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.title(f"Moving average over {configs.SB3_SMOOTH_MEAN_WINDOW} episodes")

    def trainRLWithRandomAgent(
            self,
            n_episodes: int=10
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Train the environment with a random RL agent
        """
        columns = ["action", "state", "next_state", "reward", "done", "truncated", "info"]
        results_df = Helpers.createTable(columns=columns)

        state, info = self.train_env.reset()
        for i in range(n_episodes):
            action = self.train_env.action_space.sample()
            next_state, reward, done, truncated, info = self.train_env.step(action)
            new_row = pd.Series(
                {
                    "action": action,
                    "state": state,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                    "info": info,
                })
            results_df = Helpers.appendTableRow(results_df, new_row)
            state = next_state
        Helpers.displayTable(results_df, n_rows=10, n_columns=len(columns))
        print(f"{configs.NEW_LINE}{configs.NEW_LINE}")
        prices = self.train_env.data[rl_configs.SYMBOL]
        buy_signals = info["states_sell"]
        sell_signals = info["states_buy"]
        profit = float(info["profit"])
        Helpers.plotRlBehavior(prices, buy_signals, sell_signals, profit)
        print(f"{configs.NEW_LINE}{configs.NEW_LINE}{configs.LINE_DIVIDER}{configs.NEW_LINE}")

    def reportAgentBehaviour(
            self,
            env: TradingEnv,
            info: Dict[str, Any]
    ):
        """
        Reports the agent trading behaviour after training or testing.
        :param env: TradingEnv
        :param info: Dict
        """
        prices = list(env.data[rl_configs.SYMBOL])
        buy_signals = info["states_sell"]
        sell_signals = info["states_buy"]
        profit = float(info["profit"])
        Helpers.plotRlBehavior(prices, buy_signals, sell_signals, profit)

        print(f"{configs.NEW_LINE}{configs.NEW_LINE}{configs.LINE_DIVIDER}{configs.NEW_LINE}")

    def _createAlgoTradingEnvironments(self) -> Tuple[TradingEnv, TradingEnv]:
        """
        Creates the algo trading environment
        :return: Trading train/test environment
        """
        symbol = rl_configs.SYMBOL
        features = rl_configs.FEATURES
        window = rl_configs.WINDOW
        lag = rl_configs.LAG
        data_partition_map = Helpers.getDataPartitionWindows()
        start_train = data_partition_map["train_window"]["start"]
        end_train = data_partition_map["train_window"]["end"]
        start_test = data_partition_map["test_window"]["start"]
        end_test = data_partition_map["test_window"]["end"]

        logger.info(f"Creating the 'train' RL algo trading environment..")
        train_env = TradingEnv(symbol, features, window, lag, start=start_train, end=end_train)

        logger.info(f"Creating the 'test' RL algo trading environment..")
        test_env = TradingEnv(symbol, features, window, lag, start=start_test, end=end_test)
        return train_env, test_env


