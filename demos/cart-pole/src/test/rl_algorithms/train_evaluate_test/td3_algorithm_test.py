import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.td3_algorithm import TD3TrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning import TD3HyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType

class TD3TrainAlgorithmTest(ut.TestCase):
    """
    TD3 Network Test
    """
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)
        self.env_name = "Pendulum-v1"
        self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
        self.rl_algorithm_type = RLAgorithmType.td3

    def test_TD3TrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        td3_agent = TD3TrainAlgorithm(env=self.env, rl_algorithm_type=self.rl_algorithm_type)
        self.assertIsNotNone(td3_agent, msg=error_msg)

    def test_TD3HyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = TD3HyperParameterTuning(
            self.env,
            self.rl_algorithm_type,
            rl_problem_title=self.env_name)
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_TD3TrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of TD3 RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        td3_agent = TD3TrainAlgorithm(
            self.env
        )
        self.assertIsNotNone(td3_agent, msg=error_msg)
        td3_agent.train()
        self.evaluateTrainedModel(td3_agent.trained_model)


    def test_TD3TrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the TD3 RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        td3_agent = TD3TrainAlgorithm(
            self.env
        )
        self.assertIsNotNone(td3_agent, msg=error_msg)
        td3_agent.evaluate()

    def evaluateTrainedModel(self, model):
        """
        Evaluates a trained model
        :param model: Model
        :return: None
        """
        mean_reward, std_reward = evaluate_policy(
            model,
            self.env,
            n_eval_episodes=10,
            deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

