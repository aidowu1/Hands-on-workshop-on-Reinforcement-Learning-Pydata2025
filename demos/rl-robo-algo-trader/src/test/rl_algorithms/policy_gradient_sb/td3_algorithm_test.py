import unittest as ut
import inspect
import gymnasium as gym
import os
from pprint import pprint

from src.main.rl_algorithms.policy_gradient_sb.td3_algorithm import TD3Algorithm
# from src.main.environment.env import DynamicHedgingEnv
from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers


class TD3AlgorithmTest(ut.TestCase):
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
        self.env_name = "RL Delta Hedger"
        self.env = DynamicHedgingEnv()


    def test_TD3Algorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = TD3Algorithm(env=self.env, env_name=self.env_name)
        self.assertIsNotNone(agent, msg=error_msg)

    def test_TD3Algorithm_Train_RL_Agent_Is_Valid(self):
        """
        Test the validity of training of the TD3 RL agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = TD3Algorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()

    def test_TD3Algorithm_Best_Hyperparameter_Preprocess_Is_Valid(self):
        """
        Test the validity of the best hyperparameter preprocessing for TD3 RL agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = TD3Algorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        hyper_parameters = agent.preprocessHyperParameters()
        self.assertIsNotNone(hyper_parameters, msg=error_msg)
        print("The preprocessed hyperparameters are:")
        pprint(hyper_parameters)

    def test_TD3Algorithm_Test_RL_Agent_Is_Valid(self):
        """
        Test the validity of evaluation (testing) of the TD3 RL agent post-training
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 100
        agent = TD3Algorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes
        )
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()
        agent.evaluate(n_test_episodes=n_episodes)

    def test_TD3Algorithm_Plot_Raw_Reward_Curve_Of_RL_Agent_Is_Valid(self):
        """
        Test the validity of plotting of the raw reward curve of the TD3 RL agent post-training
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = TD3Algorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes
        )
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()
        # self.env.is_run_test = True
        agent.evaluate(n_test_episodes=n_episodes)
        agent.plotRawMeanRewardCurve()

    def test_TD3Algorithm_Plot_Smooth_Reward_Curve_Of_RL_Agent_Is_Valid(self):
        """
        Test the validity of plotting of the smooth reward curve of the TD3 RL agent post-training
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = TD3Algorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes
        )
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()
        agent.evaluate(n_test_episodes=n_episodes)
        agent.plotSmoothRewardCurve()

