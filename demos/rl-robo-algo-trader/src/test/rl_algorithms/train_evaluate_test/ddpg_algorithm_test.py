import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.ddpg_algorithm import DDPGTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.ddpg_hyper_parameter_tuning import DDPGHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType

class DDPGTrainAlgorithmTest(ut.TestCase):
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
        self.env_name = "Pendulum-v1"
        self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
        self.rl_algorithm_type = RLAgorithmType.ddpg

    def test_DDPGTrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ddpg_agent = DDPGTrainAlgorithm(env=self.env, rl_algorithm_type=self.rl_algorithm_type)
        self.assertIsNotNone(ddpg_agent, msg=error_msg)

    def test_DDPGHyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = DDPGHyperParameterTuning(
            self.env,
            self.rl_algorithm_type,
            rl_problem_title=self.env_name)
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_DDPGTrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of DDPG RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ddpg_agent = DDPGTrainAlgorithm(
            self.env
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)
        ddpg_agent.train()
        self.evaluateTrainedModel(ddpg_agent.trained_model)


    def test_DDPGTrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the DDPG RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ddpg_agent = DDPGTrainAlgorithm(
            self.env
        )
        self.assertIsNotNone(ddpg_agent, msg=error_msg)
        rewards = ddpg_agent.evaluate()
        self.assertIsNotNone(rewards, msg=error_msg)

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

