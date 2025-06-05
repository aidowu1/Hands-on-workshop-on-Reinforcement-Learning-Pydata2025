import unittest as ut
import inspect
from pprint import pprint
import os

from src.main.rl_algorithms.policy_gradient_sb.ppo_algorithm import PPOAlgorithm
from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers


class PPOAlgorithmTest(ut.TestCase):
    """
    PPO Network Test
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

    def test_PPOAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the PPO RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = PPOAlgorithm(
            env=self.env,
            env_name=self.env_name)
        self.assertIsNotNone(agent, msg=error_msg)

    def test_PPOAlgorithm_Best_Hyperparameter_Preprocess_Is_Valid(self):
        """
        Test the validity of the best hyperparameter preprocessing for PPO RL agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = PPOAlgorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        hyper_parameters = agent.preprocessHyperParameters()
        self.assertIsNotNone(hyper_parameters, msg=error_msg)
        print("The preprocessed hyperparameters are:")
        pprint(hyper_parameters)

    def test_PPOAlgorithm_Train_RL_Agent_Is_Valid(self):
        """
        Test the validity of training of the PPO RL agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = PPOAlgorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()

    def test_PPOAlgorithm_Test_RL_Agent_Is_Valid(self):
        """
        Test the validity of evaluation (testing) of the PPO RL agent post-training
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = PPOAlgorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()
        agent.evaluate(n_test_episodes=n_episodes)

    def test_PPOAlgorithm_Plot_Raw_Reward_Curve_Of_RL_Agent_Is_Valid(self):
        """
        Test the validity of plotting of the raw reward curve of the PPO RL agent post-training
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = PPOAlgorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()
        agent.evaluate(n_test_episodes=n_episodes)
        agent.plotRawMeanRewardCurve()

    def test_PPOAlgorithm_Plot_Smooth_Reward_Curve_Of_RL_Agent_Is_Valid(self):
        """
        Test the validity of plotting of the smooth reward curve of the PPO RL agent post-training
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        n_episodes = 5
        agent = PPOAlgorithm(
            env=self.env,
            env_name=self.env_name,
            total_timesteps=configs2.N_STEPS * n_episodes)
        self.assertIsNotNone(agent, msg=error_msg)
        agent.train()
        agent.evaluate(n_test_episodes=n_episodes)
        agent.plotSmoothRewardCurve()

