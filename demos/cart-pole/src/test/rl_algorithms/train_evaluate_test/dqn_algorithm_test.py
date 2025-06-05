import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.dqn_algorithm import DQNTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.dqn_hyper_parameter_tuning import DQNHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType

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
        self.cartpole_name = "CartPole-v1"
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
        rewards = dqn_agent.evaluate(env=self.cart_pole_env)
        self.assertIsNotNone(rewards, msg=error_msg)

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

