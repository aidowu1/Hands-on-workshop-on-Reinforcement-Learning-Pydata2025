import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from typing import List, Tuple, Dict
import pandas as pd

import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.dqn_algorithm import DQNTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.dqn_hyper_parameter_tuning import DQNHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType
from src.main.utility.chart_results import ChartResults

class DQNTrainAlgorithmTest(ut.TestCase):
    """
    DDPG Network Test
    """
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.rl_algo_type = RLAgorithmType.dqn
        self.cartpole_name = configs.CART_POLE_PROBLEM_NAME
        self.cart_pole_env = gym.make(self.cartpole_name, render_mode="rgb_array")
        self.rl_algorithm_type = RLAgorithmType.dqn

    def test_DQNTrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the DQN RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dqn_agent = DQNTrainAlgorithm(env=self.cart_pole_env, rl_algorithm_type=self.rl_algorithm_type)
        self.assertIsNotNone(dqn_agent, msg=error_msg)

    def test_DQNHyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the DQN RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = DQNHyperParameterTuning(
            self.cart_pole_env,
            self.rl_algorithm_type,
            rl_problem_title=self.cartpole_name)
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_DQNTrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of DQN RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dqn_agent = DQNTrainAlgorithm(
            self.cart_pole_env
        )
        self.assertIsNotNone(dqn_agent, msg=error_msg)
        dqn_agent.train()
        self.evaluateTrainedModel(dqn_agent.trained_model)


    def test_DQNTrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the DQN RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dqn_agent = DQNTrainAlgorithm(
            self.cart_pole_env
        )
        self.assertIsNotNone(dqn_agent, msg=error_msg)
        env = gym.make(self.cartpole_name, render_mode="human")
        dqn_agent.evaluate(env=env)

    def test_PPOTrainAlgorithm_Plot_Reward_Curves_Agent_Model_Is_Valid(self):
        """
        Test the validity of plotting the reward curve for PPO RL trained agent.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rewards = Helpers.getSmoothedAverageRewards(
            env_name=self.cartpole_name,
            rl_algo_name=self.rl_algorithm_type.name)
        self.assertIsNotNone(rewards, msg=error_msg)
        ChartResults.plotRewardCurve(rewards, window_size=200)

    def evaluateTrainedModel(self, model):
        """
        Evaluates a trained model
        :param model: Model
        :return: None
        """
        mean_reward, std_reward = evaluate_policy(
            model,
            self.cart_pole_env,
            n_eval_episodes=10,
            deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    def _getSmoothedAverageRewards(self) -> List[float]:
        """
        Getter for the smoothed average rewards
        :return: Smoothed average rewards
        """
        results_monitor_path = f"{configs.LOG_FOLDER}/{self.cartpole_name}_{self.rl_algo_type.name}/monitor.csv"
        if os.path.exists(results_monitor_path):
            df = pd.read_csv(results_monitor_path, index_col=None, skiprows=[0])
            mean_rewards = list(df.r)
        else:
            print(f"Smoothing average rewards not found, saving to {results_monitor_path}")
            mean_rewards = []
        return mean_rewards


