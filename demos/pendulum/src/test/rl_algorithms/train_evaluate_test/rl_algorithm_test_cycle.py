import os

import gym
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional, Any
import warnings

from src.main.rl_algorithms.train_evaluate_test.base_algorithms import BaseRLAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.utility.logging import Logger
import src.main.configs.global_configs  as configs
from src.main.utility.utils import Helpers
from src.main.utility.utils import Helpers

class RLAlgorithmTestCycle:
    """
    Infrastructure to invoke the test cycles of the trained RL hedger agents
    using out-of-sample data
    """
    def __init__(
            self,
            env: gym.Env,
            agent: BaseRLAlgorithm,
            n_simulation_paths: Optional[int] = None,
            seed: int = configs.SEED,
            model_path: Optional[str] = None,
            extra_description: Optional[str] = None,
            model_use_case: str = None,
            parameters: Any = None
    ):
        """
        Constructor
        :param env: Hedger environment
        :param agent: Hedger RL agent
        :param n_simulation_paths: Number of paths (i.e., simulation paths) to run
        :param seed: Random seed
        """
        self._logger = Logger.getLogger()
        self._env = env
        self._agent = agent
        self._seed = seed
        self._algorithm = self._agent.algorithm
        self._extra_description = extra_description
        self._rl_algorithm_type = self._agent.algorithim_type
        self._problem_title = self._agent.problem_title
        self._model_use_case = model_use_case
        self._tuned_model_root_path = BaseHyperParameterTuning.createModelRootPath(
            rl_algo_type=self._rl_algorithm_type,
            problem_title=self._problem_title)
        if model_path:
            self._model_path = f"{self._agent.createSaveModelPath(custom_model_path=model_path)}.zip"
        else:
            self._model_path = f"{self._agent.createSaveModelPath()}.zip"


        self._results_folder, self._plots_folder = self.createTestResultsPath()
        self._results_path = f"{self._results_folder}/{self._hedging_type.name}.csv"
        self._plots_per_hedging_type_path = self.createDistributionPlotByHedgingTypePath()

        self._results_df = None
        self._aggregationType = AggregationType.mean
        self._pnl_agg_df = None
        self._reward_agg_df = None
        self._trading_cost_agg_df = None
        self._delta_agg_df = None

        self._single_path_index = 0
        self._pnl_single_path_agg_df = None
        self._reward_single_path_agg_df = None
        self._trading_single_path_cost_agg_df = None
        self._delta_single_path_df = None

        # Attributes for plotting results
        self._hedge_benchmark_name = self._getBenchMarkName().get("name", "BS")
        self._hedge_benchmark_delta_column_name = self._getBenchMarkName().get("delta_column_name", "bs_delta")
        self._hedge_benchmark_option_price_column_name = self._getBenchMarkName().get("option_price_column_name",
                                                                                      "bs_option_price")
        self._logger.info(f"The RL model path use in this experiment is: {self._model_path}")

    def _getBenchMarkName(self) -> Dict[str, str]:
        """
        Gets the current benchmark hedging strategy name
        :return: Map of current benchmark name
        """
        return Helpers.getHedgingBenchmarkName(self._hedging_type)

    def rlAgentTestRunSingleCycle(self, current_path: int) -> pd.DataFrame:
        """
        Test run cycle of the trained RL hedger agent for a single path/simulation
        :param current_path: Current simulation path
        :return: DataFrame containing test infos for a single episode
        """
        test_infos_enriched_df = None
        test_infos_df = None
        done = False
        try:
            if os.path.exists(self._model_path):
                self._model = self._algorithm.load(self._model_path, env=self._env)
                states, info = self._env.reset(seed=self._seed)
                while not done:
                    action_array, _states = self._model.predict(states)
                    action = action_array.item()
                    states, reward, done, _, info = self._env.step(action)

                    info["simulation_path"] = current_path

                    # Suppress the specific DeprecationWarning related to np.find_common_type
                    warnings.filterwarnings(
                        "ignore",
                        category=DeprecationWarning,
                        message=".*np.find_common_type.*",  # Use a regex pattern to match the warning message
                    )

                    if test_infos_df is None:
                        test_infos_df = pd.DataFrame(info, index=[0])
                    else:
                        test_infos_df = test_infos_df._append(info, ignore_index=True)

                test_infos_enriched_df = self.computeExtraInfosPerPath(test_infos_df)

        except Exception as ex:
            error_msg = f"Error type, {type(ex).__name__}: {ex}"
            self._logger.error(error_msg)
            raise Exception(error_msg)

        return test_infos_enriched_df

    def rlAgentTestRunAllCycles(self) -> pd.DataFrame:
        """
        Test run cycle of the trained RL hedger agent for a single path/simulation
        :return: DataFrame containing test infos for all episodes
        """
        test_all_infos_df = None
        self._logger.info(f"Running the full test cycle of the trained RL hedger agent")
        try:
            for i in tqdm(range(self._n_paths), desc="RL agent test cycle"):
                infos_df = self.rlAgentTestRunSingleCycle(i)
                # self._logger.info(f"Current iteration of path: {i}\tResult shape: {infos_df.shape}")
                if test_all_infos_df is None:
                    test_all_infos_df = infos_df
                else:
                    test_all_infos_df = test_all_infos_df._append(infos_df, ignore_index=True)
            test_all_infos_df.to_csv(self._results_path)
        except Exception as ex:
            error_msg = f"Error type, {type(ex).__name__}: {ex}"
            self._logger.error(error_msg)
            raise Exception(error_msg)

        return test_all_infos_df

    def _computeTransactionCosts(
            self,
            delta_change: float
    ):
        """
        Computes the delta hedging transaction cost
        :param delta_change: Change in delta
        :return: Delta hedging transaction cost
        """
        transaction_cost = (self._parameters.tick_size
                            * (np.abs(delta_change)
                            + 0.01 * delta_change ** 2) )
        return transaction_cost

    def _computePnl(
            self,
            option_price_change: float,
            stock_price_change: float,
            current_delta: float,
            delta_change: float,
            current_stock_price: float
    ):
        """
        Computes the Pnl
        :param option_price_change: Option price change
        :param stock_price_change: Stock price change
        :param current_delta: Current delta
        :param delta_change: Change in delta
        :param current_stock_price: Current stock price
        :return: Pnl
        """
        pnl = (option_price_change
               + (stock_price_change * current_delta)
               - (self._parameters.tick_size * np.abs(delta_change) * current_stock_price))
        return pnl

    def _computeReward(
            self,
            pnl: float,
    ):
        """
        Computes the reward
        :param pnl: Pnl
        :return: Reward
        """
        reward = pnl - (self._parameters.risk_averse_level / 2) * (pnl ** 2)
        return reward

    def computeExtraInfosPerPath(
            self,
            source_infos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the additional RL infos
        :return:
        """
        n_paths = source_infos_df.shape[0]
        source_infos_df.rl_delta = -1*source_infos_df.rl_delta
        bs_deltas = source_infos_df.bs_delta.tolist()
        rl_deltas = source_infos_df.rl_delta.tolist()
        stock_prices = source_infos_df.current_stock_price.tolist()
        option_prices = source_infos_df.current_option_price.tolist()

        bs_pnls = []
        bs_rewards = []
        bs_trading_costs = []
        rl_pnls = []
        rl_rewards = []
        rl_trading_costs = []

        for i in range(1, n_paths):
            bs_delta_change = bs_deltas[i] - bs_deltas[i - 1]
            rl_delta_change = rl_deltas[i] - rl_deltas[i - 1]
            option_price_change = option_prices[i] - option_prices[i - 1]
            stock_price_change = stock_prices[i] - stock_prices[i - 1]
            bs_pnl = self._computePnl(
                option_price_change=option_price_change,
                stock_price_change=stock_price_change,
                current_delta=bs_deltas[i],
                delta_change=bs_delta_change,
                current_stock_price=option_prices[i]
            )
            bs_trading_cost = self._computeTransactionCosts(
                delta_change=bs_delta_change
            )
            bs_reward = self._computeReward(bs_pnl)

            rl_pnl = self._computePnl(
                option_price_change=option_price_change,
                stock_price_change=stock_price_change,
                current_delta=rl_deltas[i],
                delta_change=rl_delta_change,
                current_stock_price=option_prices[i]
            )
            rl_trading_cost = self._computeTransactionCosts(
                delta_change=rl_delta_change
            )
            rl_reward = self._computeReward(rl_pnl)

            bs_pnls.append(bs_pnl)
            bs_trading_costs.append(bs_trading_cost)
            bs_rewards.append(bs_reward)

            rl_pnls.append(rl_pnl)
            rl_trading_costs.append(rl_trading_cost)
            rl_rewards.append(rl_reward)

        last_n_rows = n_paths - 1
        target_infos_df = source_infos_df.copy().tail(last_n_rows)
        target_infos_df["bs_pnl"] = bs_pnls
        target_infos_df["bs_trading_cost"] = bs_trading_costs
        target_infos_df["bs_reward"] = bs_rewards

        target_infos_df["rl_pnl"] = rl_pnls
        target_infos_df["rl_trading_cost"] = rl_trading_costs
        target_infos_df["rl_reward"] = rl_rewards

        column_to_move = target_infos_df.pop("simulation_path")
        target_infos_df.insert(0, "simulation_path", column_to_move)

        return target_infos_df

    def aggregateResults(
            self,
            aggregation_type: AggregationType
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aggregate the results
        :return: Aggregated results
        """
        if os.path.exists(self._results_path):
            self._results_df = pd.read_csv(self._results_path, index_col=False)
            self._aggregationType = aggregation_type

            # Suppress the specific DeprecationWarning related to np.find_common_type
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=".*np.find_common_type.*",  # Use a regex pattern to match the warning message
            )

            if aggregation_type is AggregationType.mean:
                pnl_cols = ["simulation_path", "bs_pnl", "rl_pnl"]
                pnl_df = self._results_df[pnl_cols]
                self._pnl_agg_df = pnl_df.groupby(pnl_cols.pop(0))[pnl_cols].mean()

                trading_cost_cols = ["simulation_path", "bs_trading_cost", "rl_trading_cost"]
                trading_cost_df = self._results_df[trading_cost_cols]
                self._trading_cost_agg_df = trading_cost_df.groupby(trading_cost_cols.pop(0))[trading_cost_cols].mean()

                reward_cols = ["simulation_path", "bs_reward", "rl_reward"]
                reward_df = self._results_df[reward_cols]
                self._reward_agg_df = reward_df.groupby(reward_cols.pop(0))[reward_cols].mean()

                delta_cols = ["simulation_path", "bs_delta", "rl_delta"]
                delta_df = self._results_df[delta_cols]
                self._delta_agg_df = delta_df.groupby(delta_cols.pop(0))[delta_cols].mean()
            else:
                pnl_cols = ["simulation_path", "bs_pnl", "rl_pnl"]
                pnl_df = self._results_df[pnl_cols]
                self._pnl_agg_df = pnl_df.groupby(pnl_cols.pop(0))[pnl_cols].sum()

                trading_cost_cols = ["simulation_path", "bs_trading_cost", "rl_trading_cost"]
                trading_cost_df = self._results_df[trading_cost_cols]
                self._trading_cost_agg_df = trading_cost_df.groupby(trading_cost_cols.pop(0))[trading_cost_cols].sum()

                reward_cols = ["simulation_path", "bs_reward", "rl_reward"]
                reward_df = self._results_df[reward_cols]
                self._reward_agg_df = reward_df.groupby(reward_cols.pop(0))[reward_cols].sum()

                delta_cols = ["simulation_path", "bs_delta", "rl_delta"]
                delta_df = self._results_df[delta_cols]
                self._delta_agg_df = delta_df.groupby(delta_cols.pop(0))[delta_cols].mean()
        else:
            self._logger.warn(f"{self._results_path} does not exist..")
        return self._pnl_agg_df, self._reward_agg_df, self._trading_cost_agg_df, self._delta_agg_df

    def getSinglePathResults(
            self,
            test_path_index:int =None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Gets the RL results for a single path evaluation
        :param test_path_index: Path
        :return: Single path results
        """
        self._single_path_index = test_path_index
        if test_path_index is None:
            self._single_path_index = np.random.randint(self._n_paths)
        else:
            self._single_path_index = test_path_index

        self._logger.info(f"Testing the {self._rl_algorithm_type} RL agent for Simulation path: {test_path_index}")
        if os.path.exists(self._results_path):
            self._results_df = pd.read_csv(self._results_path, index_col=False)
            episode_filter = self._results_df.simulation_path_index == self._single_path_index
            episode_results_df = self._results_df[episode_filter].reset_index()

            # Suppress the specific DeprecationWarning related to np.find_common_type
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=".*np.find_common_type.*",  # Use a regex pattern to match the warning message
            )

            pnl_cols = ["bs_pnl", "rl_pnl"]
            self._pnl_single_path_df = episode_results_df[pnl_cols]

            trading_cost_cols = ["bs_trading_cost", "rl_trading_cost"]
            self._trading_cost_single_path_df = episode_results_df[trading_cost_cols]

            reward_cols = ["bs_reward", "rl_reward"]
            self._reward_single_path_df = episode_results_df[reward_cols]

            delta_cols = ["bs_delta", "rl_delta"]
            self._delta_single_path_df = episode_results_df[delta_cols]
        return (self._pnl_single_path_df, self._trading_cost_single_path_df,
                self._reward_single_path_df, self._delta_single_path_df)

    def createTestResultsPath(self) -> Tuple[str, str]:
        """
        Creates the test results path
        :return: Test results path
        """
        joined_title = "_".join(self._problem_title.split())
        # root_path = f"./logs/{joined_title}_{self._rl_algorithm_type.name}_{self._hedging_type.name}"
        root_path = f"{self._tuned_model_root_path}/{self._hedging_type.name}"
        if self._extra_description:
            results_path = f"{root_path}/test_results/{self._extra_description}"
            plots_path = f"{root_path}/plots/{self._extra_description}"
        else:
            results_path = f"{root_path}/test_results"

            plots_path = f"{root_path}/plots"
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)
        return results_path, plots_path
    def createDistributionPlotByHedgingTypePath(self) -> str:
        """
        Creates the RL comparative results plot path
        :return: Test results path
        """
        joined_title = "_".join(configs2.RL_PROBLEM_TITLE.split())
        if self._extra_description:
            root_path = f"{self._tuned_model_root_path}/perf_comparative_results/{self._extra_description}"
            plots_path = f"{root_path}/{self._hedging_type.name}/{self._extra_description}"
        else:
            root_path = f"{self._tuned_model_root_path}/perf_comparative_results"
            plots_path = f"{root_path}/{self._hedging_type.name}"
        os.makedirs(plots_path, exist_ok=True)
        return plots_path

    def plotTwoVariableEvaluationResults(
            self,
            plot_type: PlotType = PlotType.rewards,
            is_single_episode = False
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
        if plot_type is PlotType.rewards:
            plt.plot(self._reward_agg_df.rl_reward.tolist(), label='RL')
            plt.plot(self._reward_agg_df.bs_reward.tolist(), label=self._hedge_benchmark_name)
            plt.ylabel(f"Reward ({self._aggregationType.name})")
        elif plot_type is PlotType.trading_cost:
            plt.plot(self._trading_cost_agg_df.rl_trading_cost.tolist(), label='RL')
            plt.plot(self._trading_cost_agg_df.bs_trading_cost.tolist(), label=self._hedge_benchmark_name)
            plt.ylabel(f"Trading Cost ({self._aggregationType.name})")
        elif plot_type is PlotType.pnl:
            plt.plot(self._pnl_agg_df.rl_pnl.tolist(), label='RL')
            plt.plot(self._pnl_agg_df.bs_pnl.tolist(), label=self._hedge_benchmark_name)
            plt.ylabel(f"Pnl ({self._aggregationType.name})")
        elif plot_type is PlotType.delta:
            plt.plot(self._delta_agg_df.rl_delta.tolist(), label='RL')
            plt.plot(self._delta_agg_df.bs_delta.tolist(), label=self._hedge_benchmark_name)
            plt.ylabel("Delta (mean)")
        else:
            raise Exception("Invalid plot_type!")
        if not is_single_episode:
            plt.xlabel("Episodes")
        else:
            plt.xlabel("Time steps")
        plt.legend()
        # plt.show()
        plot_path = f"{self._plots_folder}/{plot_type.name}.png"
        plt.savefig(plot_path)
        plt.close()

    def plotTwoVariableKernelDesityEstimations(
            self,
            plot_type: PlotType = PlotType.delta,
    ):
        """
        Plots PnL kernel estimation densities for a specified RL algorithm
        :param plot_type: Plot type
        """
        plt.figure(figsize=(12, 6))
        if plot_type is PlotType.pnl:
            sns.kdeplot(data=self._pnl_agg_df)
            plt.xlabel(f"Pnl ({self._aggregationType.name})")
            plt.title(f'{self._hedge_benchmark_name} versus RL PnL {self._aggregationType.name} for '
                      f'{self._rl_algorithm_type.name} RL agent')
        elif plot_type is PlotType.rewards:
            sns.kdeplot(data=self._reward_agg_df)
            plt.xlabel(f"Reward ({self._aggregationType.name})")
            plt.title(f"{self._hedge_benchmark_name} versus RL Rewards {self._aggregationType.name} for "
                      f"{self._rl_algorithm_type.name} RL agent")
        elif plot_type is PlotType.trading_cost:
            sns.kdeplot(data=self._trading_cost_agg_df)
            plt.xlabel(f"Trading Cost ({self._aggregationType.name})")
            plt.title(f"{self._hedge_benchmark_name} versus RL Trading Cost {self._aggregationType.name} "
                      f"for {self._rl_algorithm_type.name} RL agent")
        elif plot_type is PlotType.delta:
            sns.kdeplot(data=self._delta_agg_df)
            plt.xlabel("Delta (mean)")
            plt.title(f"{self._hedge_benchmark_name} versus RL delta {self._aggregationType.name} "
                      f"for {self._rl_algorithm_type.name} RL agent")
        else:
            raise Exception("Invalid plot_type!")

        plot_path = f"{self._plots_folder}/{plot_type.name}_density.png"
        plt.savefig(plot_path)
        plt.close()

    def plotTwoVariableKernelDesityEstimationsAllPlots(
            self,
            plot_name: str = "all_distributions",
            is_plot_2_screen: bool = False
    ):
        """
        Plots PnL kernel estimation densities for all RL algorithms
        :param plot_name: Plot name
        :param is_plot_2_screen: Flag to indicate plotting to screen
        """
        fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=False)
        fig.suptitle(f"Delta hedging {self._hedge_benchmark_name} benchmark versus {self._rl_algorithm_type.name.upper()} ")
        # fig.tight_layout()

        axes[0].set_title(f"a) RL agent hedging for {self._aggregationType.name} PnL",
                          x=0.02, y=0.9, pad=-1, loc="left", fontdict={'fontsize': 10})
        sns.kdeplot(ax=axes[0], data=self._pnl_agg_df)
        # axes[0].legend()
        axes[0].grid(True)

        axes[1].set_title(f"b) RL agent hedging for {self._aggregationType.name} Rewards",
                          x=0.02, y=0.9, pad=-1, loc="left", fontdict={'fontsize': 10})
        sns.kdeplot(ax=axes[1], data=self._reward_agg_df)
        # axes[1].legend()
        axes[1].grid(True)

        axes[2].set_title(f"c) RL agent hedging for {self._aggregationType.name} Trading Cost",
                          x=0.02, y=0.9, pad=-1, loc="left", fontdict={'fontsize': 10})
        sns.kdeplot(ax=axes[2], data=self._trading_cost_agg_df)
        axes[2].grid(True)

        axes[3].set_title(f"d) RL agent hedging for {self._aggregationType.name} Delta",
                          x=0.02, y=0.9, pad=-1, loc="left", fontdict={'fontsize': 10})
        sns.kdeplot(ax=axes[3], data=self._delta_agg_df)
        axes[3].grid(True)

        axes[4].set_title(f"e) RL agent Delta hedge for simulation path index @ {self._single_path_index}",
                          x=0.02, y=0.9, pad=-1, loc="left", fontdict={'fontsize': 10})
        sns.lineplot(ax=axes[4], data=self._delta_single_path_df)
        axes[4].grid(True)

        axes[5].set_title(f"f) RL agent Reward for simulation path index @ {self._single_path_index}",
                          x=0.02, y=0.9, pad=-1, loc="left", fontdict={'fontsize': 10})
        sns.lineplot(ax=axes[5], data=self._reward_single_path_df)
        axes[5].grid(True)

        plt.legend(loc='best', shadow=True)
        plt.subplots_adjust(hspace=0.4)
        plot_path = f"{self._plots_per_hedging_type_path}/{self._rl_algorithm_type.name}_{plot_name}.png"
        print(f"Saving the plot for {self._rl_algorithm_type.name} hedger agent at the path: {plot_path}")
        plt.savefig(plot_path)
        if is_plot_2_screen:
            plt.show()
        plt.close()


    @property
    def results_df(self) -> pd.DataFrame:
        """
        Getter for the test results dataframe
        :return: Test results dataframe
        """
        return self._results_df

    @property
    def results_path(self) -> str:
        """
        Getter of the results path
        :return: Results path
        """
        return self._results_path



