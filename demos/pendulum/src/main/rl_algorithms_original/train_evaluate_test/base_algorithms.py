import os
import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, DDPG, SAC, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
import os
from abc import ABC, abstractmethod

from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.market_simulator.caching import SimulationDataCache
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.utility.logging import Logger
from src.main.utility.enum_types import PlotType

class BaseRLAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    """
    def __init__(
            self,
            env: DynamicHedgingEnv,
            rl_algorithm_type: RLAgorithmType,
            hedging_type: HedgingType,
            rl_problem_title: str ="RL Delta Hedger",
            total_timesteps: int = configs2.N_TUNING_TRAIN_STEPS,
            check_freq: int = 100,
            model_use_case: str = None
    ):
        """
        Constructor
        :param env: Environment
        """
        self._logger = Logger().getLogger()
        self._policy_name = "MlpPolicy"
        self._rl_problem_title = rl_problem_title
        self._hedging_type = hedging_type
        self._total_timesteps = total_timesteps
        self._check_freq = check_freq
        self._rl_algo_type = rl_algorithm_type
        self._model_use_case = model_use_case
        self._log_dir = self.createLogPath()
        self._tuned_model_root_path = BaseHyperParameterTuning.createModelRootPath(
            rl_algo_type=self._rl_algo_type,
            model_use_case=self._model_use_case)
        self._model_path = self.createSaveModelPath()
        self._plot_dir = self.createPlotPath()
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._plot_dir, exist_ok=True)
        self._tensorboard_dir = f"tensorboard/{self._rl_problem_title}_{self._rl_algo_type.name}_{self._hedging_type.name}"
        self._env = Monitor(env, self._log_dir)
        self._n_simulation_time_steps = env.n_simulation_time_steps
        self._seed = configs2.SEED

        # Evaluation variables
        self._simulation_path_index = configs2.DEFAULT_PATH_INDEX
        self._current_transaction_cost = []
        self._current_pnl = []
        self._replication_portfolio_value = []
        self._money_market_account = []
        self._bs_delta = []
        self._rl_delta = []
        self._current_stock_price = []
        self._current_option_price = []
        self._rewards = []
        self._evaluation_results_df = pd.DataFrame()
        self._rl_algorithms = {
            RLAgorithmType.ddpg: DDPG,
            RLAgorithmType.sac: SAC,
            RLAgorithmType.td3: TD3,
            RLAgorithmType.ppo: PPO
        }
        self._model = None
        self._delta_df = None
        self._option_price_df = None
        self._reward_df = None

        # Attributes for plotting results
        self._hedge_benchmark_name = self._getBenchMarkName().get("name", "BS")
        self._hedge_benchmark_delta_column_name = self._getBenchMarkName().get("delta_column_name", "bs_delta")
        self._hedge_benchmark_option_price_column_name = self._getBenchMarkName().get("option_price_column_name",
                                                                                      "bs_option_price")

    @abstractmethod
    def train(self):
        """
        Trains the RL algorithm.
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def tuned_hyper_parameters(self) -> Dict[str, Any]:
        """
        Gets and pre-processes the best tuned hyperparameters for the RL algorithm
        :return: Best hyperparameters
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def non_tuned_hyperparameters(self) -> Dict[str, Any]:
        """
        Getter for the non-turned hyperparameters
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def hyper_parameters(self) -> Dict[str, Any]:
        """
        Getter for RL algorithm hyperparameters
        :return: RL algorithm hyperparameters
        """
        if configs2.IS_USE_HYPER_PARAMETER_TUNING:
            self._logger.info("Training is using tuned hyperparameters...")
            self._logger.info(f"Hyperparameters are:\n{self.tuned_hyper_parameters}")
            return self.tuned_hyper_parameters
        else:
            self._logger.info("Training is using non-tuned hyperparameters...")
            self._logger.info(f"Hyperparameters are:\n{self.non_tuned_hyperparameters}")
            return self.non_tuned_hyperparameters

    def evaluate(
            self,
            model_path: str = None
    ) -> pd.DataFrame:
        """
        Evaluates the RL algorithm.
        :param model_path: Model path
        :param selected_path_index: Selected simulation path index
        :return: Evaluation results
        """
        if model_path is None:
            model_path = self._model_path
        algorithm = self._rl_algorithms[self._rl_algo_type]
        self._model = algorithm.load(model_path, env=self._env)
        states, info = self._env.reset(seed=self._seed)

        for _ in tqdm(range(self._n_simulation_time_steps), desc="Evaluating RL Algorithm.."):
            action_array, _states = self._model.predict(states)
            action = action_array.item()
            states, reward, done, _, info = self._env.step(action)
            self._simulation_path_index = info["simulation_path_index"]
            self._current_transaction_cost.append(info["current_transaction_cost"])
            self._current_pnl.append(info["current_pnl"])
            self._replication_portfolio_value.append(info["hedge_portfolio_value"])
            self._money_market_account.append(info["money_market_account"])
            self._bs_delta.append(info["bs_delta"])
            self._current_stock_price.append(info["current_stock_price"])
            self._current_option_price.append(info["current_option_price"])
            self._rewards.append(reward)
            self._rl_delta.append(action)

            if done:
                break

        self._createEvaluationResultsTable()
        self._createEvaluationSubResults()
        self._logger.info(f"Created evaluation results for {self._rl_algo_type.name} with {self._hedging_type.name} "
                          f"hedging type")
        self._logger.info(f"The results table has columns and rows of {self._evaluation_results_df.shape[1]} "
                          f"and {self._evaluation_results_df.shape[0]} respectively")
        return self._evaluation_results_df


    def _createEvaluationResultsTable(self):
        """
        Creates the evaluation results table.
        :return: Evaluation results table
        """
        evaluation_results = {
            "current_transaction_cost": self._current_transaction_cost,
            "current_pnl": self._current_pnl,
            "rl_option_price": self._replication_portfolio_value,
            "money_market_account": self._money_market_account,
            "bs_delta": self._bs_delta,
            "rl_delta": [-s for s in self._rl_delta],
            "current_stock_price": self._current_stock_price,
            "bs_option_price": self._current_option_price,
            "rewards": self._rewards,
        }
        self._evaluation_results_df = pd.DataFrame.from_dict(evaluation_results)

    def createModel(self) -> BaseAlgorithm:
        """
        Creates the RL algorithm/model
        :return: Returns the RL algorithm
        """
        algorithm = self._rl_algorithms[self._rl_algo_type]
        match self._rl_algo_type:
            case RLAgorithmType.td3 | RLAgorithmType.ddpg:
                return algorithm(
                            self._policy_name,
                            self._env,
                            tensorboard_log=self._tensorboard_dir,
                            **self.hyper_parameters
                        )
            case RLAgorithmType.sac | RLAgorithmType.ppo:
                return algorithm(
                    self._policy_name,
                    self._env,
                    tensorboard_log=self._tensorboard_dir,
                    **self.hyper_parameters
                )
            case _:
                raise Exception("Invalid RL algorithm type!!")

    def plotRawRewardCurve(
            self,
            log_path: str,
            plot_title: str = "RL Delta Hedger",
            time_steps: int = 1E5
    ):
        """
        Plots the raw reward curve
        :param log_path: Path to log file
        :param plot_title: Plot title
        :return:
        """
        # Helper from the library
        results_plotter.plot_results(
            [log_path], time_steps, results_plotter.X_TIMESTEPS, plot_title
        )

    def _computeMovingAverage(
            self,
            values: np.ndarray,
            window: int):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")

    def plotSmoothRewardCurve(
            self,
            log_path: str,
            plot_title: str ="Learning Curve",
            filter_window: int =configs2.PLOT_FILTER_WINDOW):
        """
        plot the results
        :param log_path: the save location of the results to plot
        :param title: the title of the task to plot
        :filter_window: the filter window to plot
        """
        x, y = ts2xy(load_results(log_path), "timesteps")
        y = self._computeMovingAverage(y, window=filter_window)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(plot_title)
        plt.plot(x, y)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.title(plot_title + " Smoothed")
        plt.show()

    def plotRewardEvaluationResults(
            self,
            is_cumulative_reward: bool = True):
        """
        Plots the cumulative reward curve
        :param is_cumulative_reward: (bool)
        :return: None
        """
        plt.figure(figsize=(12, 6))
        if is_cumulative_reward:
            plt.plot(np.cumsum(self._rewards))
            plt.ylabel('Cumulative Rewards')
        else:
            plt.plot(self._rewards)
            plt.ylabel('Rewards')
        plt.xlabel('step')

        plt.show()

    def plotTwoVariableEvaluationResults(
            self,
            plot_type: PlotType = PlotType.delta,
    ):
        """
        Plots a 2-variable evaluation result curves
        The variables are any of the following use cases:
            - Reinforcement Learning (RL) and Black Scholes (BS) delta
            - Replication portfolio (synthetic option) and actual Call option
        :param plot_type: Plot type
        :return: None
        """
        plt.figure(figsize=(12, 6))
        if plot_type is PlotType.delta:

            plt.plot(-np.array(self._rl_delta), label='RL delta')
            plt.plot(self._bs_delta, label=f'{self._hedge_benchmark_name} delta')
            plt.ylabel('Delta')
        elif plot_type is PlotType.option_price:
            plt.plot(self._replication_portfolio_value, label=f'{self._hedge_benchmark_name} Hedge portfolio value')
            plt.plot(self._current_option_price, label='Call price')
            plt.ylabel('Value')
        plt.xlabel('Time step')

        plt.legend()
        plt.show()

    def plotTwoVariableKernelDesityEstimations(
            self,
            evaluation_result_df: pd.DataFrame,
            plot_type: PlotType = PlotType.delta,
    ):
        """
        Plots PnL kernel estimation desities for a specified RL algorithm
        :param evaluation_result_df: Evaluation result dataframe
        :param plot_type: Plot type
        """

        plt.figure(figsize=(12, 6))
        if plot_type is PlotType.delta:
            selected_columns = ["bs_delta", "rl_delta"]
            data_df = evaluation_result_df[selected_columns]
            data_df.rl_delta = -1*data_df.rl_delta
            data_df.rename({
                "bs_delta": self._hedge_benchmark_delta_column_name,
            }, inplace=True)
            sns.kdeplot(data=data_df)
            plt.xlabel('Time steps')
            plt.title(f'BS versus RL Option price distribution plots for {self._rl_algo_type} RL agent')
        elif plot_type is PlotType.option_price:
            selected_columns = ["bs_option_price", "rl_option_price"]
            data_df = evaluation_result_df[selected_columns]
            data_df.rename({
                "bs_option_price": self._hedge_benchmark_option_price_column_name,
            }, inplace=True)
            sns.kdeplot(data=data_df)
            plt.xlabel('Time steps')
            plt.title(f'BS versus RL Option delta distribution plots for {self._rl_algo_type} RL agent')
        else:
            raise Exception("Invalid plot_type!")
        plt.show()

    def _createEvaluationSubResults(self):
        """
        Get subset of the evaluation result dataframe
        :return:
        """
        option_price_cols = ["bs_option_price", "rl_option_price"]
        self._option_price_df = self._evaluation_results_df[option_price_cols]

        delta_cols = ["bs_delta", "rl_delta"]
        self._delta_df = self._evaluation_results_df[delta_cols]

    def createEvaluationPlots(
            self,
            is_plot_to_screen: bool = False,
            plot_name: str = ""
    ):
        """
        Plots PnL kernel estimation densities for all RL algorithms
        :param is_plot_to_screen: Flag to indicate plotting to screen
        :param plot_name: Plot name
        """
        plot_name = f"all_distributions_{plot_name}"
        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
        hedging_type_name = "bs" if self._hedging_type is HedgingType.gbm else self._hedging_type.name

        axes[0].set_title(f"a) {self._hedge_benchmark_name} benchmark versus {self._rl_algo_type.name.upper()} "
                  f"RL agent hedging for {hedging_type_name} Option Price (density plot) "
                  f"with simulation path index: {self._simulation_path_index}", y=0.7, pad=-20, fontsize=9)
        sns.kdeplot(ax=axes[0], data=self._option_price_df)
        axes[0].grid(True)

        axes[1].set_title(f"b) {self._hedge_benchmark_name} benchmark versus {self._rl_algo_type.name.upper()} "
                f"RL agent hedging for {hedging_type_name} Deltas (density plot) "
                f"with simulation path index: {self._simulation_path_index}", y=0.7, pad=-20, fontsize=9)
        sns.kdeplot(ax=axes[1], data=self._delta_df)
        axes[1].grid(True)

        axes[2].set_title(f"c) {self._hedge_benchmark_name} benchmark versus {self._rl_algo_type.name.upper()} "
                f"RL agent hedging for {hedging_type_name} Option Price "
                f"with simulation path index: {self._simulation_path_index}", y=1.2, pad=-20, fontsize=9)
        sns.lineplot(ax=axes[2], data=self._option_price_df)
        axes[2].grid(True)

        axes[3].set_title(f"d) {self._hedge_benchmark_name} benchmark versus {self._rl_algo_type.name.upper()} "
                f"RL agent hedging for {hedging_type_name} Deltas "
                f"with simulation path index: {self._simulation_path_index}", y=1.2, pad=-20, fontsize=9)
        sns.lineplot(ax=axes[3], data=self._delta_df)
        axes[3].grid(True)

        plt.legend(loc='best', shadow=True)
        plt.subplots_adjust(hspace=0.4)
        plot_path = f"{self._plot_dir}/{self._rl_algo_type.name}_{plot_name}.png"
        print(f"Saving the plot for {self._rl_algo_type.name} hedger agent at the path: {plot_path}")
        plt.savefig(plot_path)
        if is_plot_to_screen:
            plt.show()
        plt.close()

    def _getBenchMarkName(self) -> Dict[str, str]:
        """
        Gets the current benchmark hedging strategy name
        :return: Map of current benchmark name
        """
        return Helpers.getHedgingBenchmarkName(self._hedging_type)
    def createLogPath(self) -> str:
        """
        Creates the log path
        :return: Log path
        """
        joined_title = "_".join(self._rl_problem_title.split())
        log_path = f"./logs/{joined_title}_{self._rl_algo_type.name}_{self._hedging_type.name}"
        return log_path

    def createPlotPath(self) -> str:
        """
        Creates the plot path
        :return: Plot path
        """
        joined_title = "_".join(self._rl_problem_title.split())
        plot_path = f"./Plots/{joined_title}_{self._rl_algo_type.name}_{self._hedging_type.name}"
        return plot_path

    def createSaveModelPath(
            self,
            custom_model_path: str = None
    ) -> str:
        """
        Creates the save model path
        :return: Model save path
        """
        # test_case_pattern = configs2.TUNED_TEST_USE_CASE.format(self._hedging_type.name,self._rl_algo_type.name)
        # tuned_model_root_path = f"{configs2.TUNED_MODEL_PATH}/{self._rl_algo_type.name}/{test_case_pattern}"
        if custom_model_path is None:
            model_path = f"{self._tuned_model_root_path}best_model"
        else:
            model_path = f"{custom_model_path}/best_model"
        return model_path

    @property
    def trained_model(self) -> BaseAlgorithm:
        """
        Getter for the trained model
        :return: Trained model
        """
        return self._model

    @property
    def algorithm(self) -> Any:
        """
        Getter for the trained model
        :return: Trained model
        """
        return self._rl_algorithms[self._rl_algo_type]

    @property
    def problem_title(self) -> str:
        """
        Getter for the problem title
        :return: Problem title
        """
        return self._rl_problem_title

    @property
    def algorithim_type(self) -> RLAgorithmType:
        """
        Getter for the algorithm type
        :return: Algorithm type
        """
        return self._rl_algo_type

    @property
    def hedging_type(self) -> HedgingType:
        """
        Getter for the hedging type
        :return: Hedging type
        """
        return self._hedging_type

    @property
    def evaluation_results_df(self):
        """
        Getter for the evaluation results dataframe
        :return: Evaluation results dataframe
        """
        return self._evaluation_results_df

    @property
    def evaluation_sub_results_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Getter for evaluation sub-results dataframe
        :return: Sub-results dataframe
        """
        return (self._option_price_df, self._delta_df)

    @property
    def simulation_path_index(self) -> int:
        """
        Getter for the simulation path index
        :return: Simulation path index
        """
        return self._simulation_path_index

