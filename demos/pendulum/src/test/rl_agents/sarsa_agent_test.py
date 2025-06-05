import unittest as ut
import inspect
import os
from dataclasses import asdict

from src.main.rl_agents.sarsa_agent import SarsaAgent
from src.main.configs.q_learning_agent_configs import QLearningAgentConfig
import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.utility.chart_results import ChartResults


class QLearningAgentTest(ut.TestCase):
    """
    Test suit for QLearningAgent
    """
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

    def test_QLearning_Agent_Constructor_Is_Valid(self):
        """
        Test the validity of QLearningAgent constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        configs = asdict(QLearningAgentConfig())
        sarsa_agent = SarsaAgent(**configs)
        self.assertTrue(isinstance(sarsa_agent, SarsaAgent), msg=error_msg)
        self.assertIsNotNone(sarsa_agent, msg=error_msg)

    def test_QLearning_Agent_Run_Is_Valid(self):
        """
        Test the validity of QLearningAgent run/execution
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        configs = asdict(QLearningAgentConfig())
        sarsa_agent = SarsaAgent(**configs)
        self.assertTrue(isinstance(sarsa_agent, SarsaAgent), msg=error_msg)
        self.assertIsNotNone(sarsa_agent, msg=error_msg)
        sarsa_agent.train()

    def test_QLearning_Agent_With_Reward_Curve_Plot_Is_Valid(self):
        """
        Test the validity of QLearningAgent run/execution with reward curve plot
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent_configs = asdict(QLearningAgentConfig())
        sarsa_agent = SarsaAgent(**agent_configs)
        self.assertTrue(isinstance(sarsa_agent, SarsaAgent), msg=error_msg)
        self.assertIsNotNone(sarsa_agent, msg=error_msg)
        cum_reward = sarsa_agent.train()
        Helpers.serialObject(sarsa_agent, configs.FROZEN_LAKE_SARSA_MODEL_FILE_PATH)
        ChartResults.plotRewardCurve(cum_reward, window_size=200)

