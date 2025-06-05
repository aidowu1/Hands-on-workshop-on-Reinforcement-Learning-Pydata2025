import unittest as ut
import inspect
import os
import gymnasium as gym

from src.main.rl_agents.q_learning_agent import QLearningAgent
from src.main.configs.q_learning_agent_configs import QLearningAgentConfig
import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.utility.chart_results import ChartResults
from src.main.evaluations.evaluate_agent import EvaluateAgent


class EvaluateAgentTest(ut.TestCase):
    """
    Test suit for Evaluate Agents
    """
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

    def test_Evaluate_Agent_Constructor_Is_Valid(self):
        """
        Test the validity of EvaluateAgent constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = Helpers.deserializeObject(configs.FROZEN_LAKE_QL_MODEL_FILE_PATH)
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
        rl_evaluator = EvaluateAgent(agent=agent, env=env, render_mode="human")
        self.assertIsNotNone(rl_evaluator, msg=error_msg)

    def test_Evaluate_Agent_Q_Learning_RL_Is_Valid(self):
        """
        Test the validity of Q-Learning RL agent
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = Helpers.deserializeObject(configs.FROZEN_LAKE_QL_MODEL_FILE_PATH)
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
        rl_evaluator = EvaluateAgent(agent=agent, env=env, render_mode="human")
        self.assertIsNotNone(rl_evaluator, msg=error_msg)
        rl_evaluator.run()

    def test_Evaluate_Agent_Sarsa_RL_Is_Valid(self):
        """
        Test the validity of SARSA RL agent
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = Helpers.deserializeObject(configs.FROZEN_LAKE_SARSA_MODEL_FILE_PATH)
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
        rl_evaluator = EvaluateAgent(agent=agent, env=env, render_mode="human")
        self.assertIsNotNone(rl_evaluator, msg=error_msg)
        rl_evaluator.run()