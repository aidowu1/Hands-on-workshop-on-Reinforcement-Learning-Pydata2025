import unittest as ut
import inspect
import os
from dataclasses import asdict

from src.main.rl_agents.q_learning_agent import QLearningAgent
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
        q_learning_agent = QLearningAgent(**configs)
        self.assertTrue(isinstance(q_learning_agent, QLearningAgent), msg=error_msg)

    def test_QLearning_Agent_Run_Is_Valid(self):
        """
        Test the validity of QLearningAgent run/execution
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        configs = asdict(QLearningAgentConfig())
        q_learning_agent = QLearningAgent(**configs)
        self.assertTrue(isinstance(q_learning_agent, QLearningAgent), msg=error_msg)
        q_learning_agent.run()

    def test_QLearning_Agent_With_Reward_Curve_Plot_Is_Valid(self):
        """
        Test the validity of QLearningAgent run/execution with reward curve plot
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent_configs = asdict(QLearningAgentConfig())
        q_learning_agent = QLearningAgent(**agent_configs)
        self.assertTrue(isinstance(q_learning_agent, QLearningAgent), msg=error_msg)
        _, cum_reward, _ = q_learning_agent.run()
        Helpers.serialObject(q_learning_agent, configs.GRIDWORLD_QL_MODEL_FILE_PATH)
        ChartResults.plotRewardCurve(cum_reward, window_size=200)

