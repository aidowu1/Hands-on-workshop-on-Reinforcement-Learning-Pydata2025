import unittest as ut
import inspect
import os
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.sac_algorithm import SACTrainAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.sac_hyper_parameter_tuning import SACHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType

class SACTrainAlgorithmTest(ut.TestCase):
    """
    SAC Network Test
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
        self.rl_algorithm_type = RLAgorithmType.sac

    def test_SACTrainAlgorithm_Constructor_Is_Valid(self):
        """
        Test the validity of constructing the SAC RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        sac_agent = SACTrainAlgorithm(env=self.env, rl_algorithm_type=self.rl_algorithm_type)
        self.assertIsNotNone(sac_agent, msg=error_msg)

    def test_SACHyperParameterTuning_Hyper_Parameter_Tuning_Is_Valid(self):
        """
        Test the validity of constructing the SAC RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        hyper_param_tuner = SACHyperParameterTuning(
            self.env,
            self.rl_algorithm_type,
            rl_problem_title=self.env_name)
        self.assertIsNotNone(hyper_param_tuner, msg=error_msg)
        hyper_param_tuner.run()

    def test_SACTrainAlgorithm_Train_Agent_Model_Is_Valid(self):
        """
        Test the validity of training of SAC RL algorithm.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        sac_agent = SACTrainAlgorithm(
            self.env
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        sac_agent.train()
        self.evaluateTrainedModel(sac_agent.trained_model)


    def test_SACTrainAlgorithm_Evaluate_Trained_Agent_Is_Valid(self):
        """
        Test the validity of evaluation of the SAC RL trained agent.
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        sac_agent = SACTrainAlgorithm(
            self.env
        )
        self.assertIsNotNone(sac_agent, msg=error_msg)
        sac_agent.evaluate()

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

