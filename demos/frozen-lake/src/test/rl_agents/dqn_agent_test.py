import unittest as ut
import inspect
import os
import gymnasium as gym

from src.main.rl_agents.dqn_agent import DqnAgent
import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.utility.chart_results import ChartResults


class DqnAgentTest(ut.TestCase):
    """
    Test suit for DqnAgent
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
        Test the validity of DqnAgent constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
        dqn_agent = DqnAgent(env)
        self.assertTrue(isinstance(dqn_agent, DqnAgent), msg=error_msg)
        self.assertIsNotNone(DqnAgent, msg=error_msg)

    def test_QLearning_Agent_Run_Is_Valid(self):
        """
        Test the validity of DqnAgent run/execution
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
        dqn_agent = DqnAgent(env)
        self.assertTrue(isinstance(dqn_agent, DqnAgent), msg=error_msg)
        self.assertIsNotNone(DqnAgent, msg=error_msg)
        dqn_agent.train()

    def test_QLearning_Agent_With_Reward_Curve_Plot_Is_Valid(self):
        """
        Test the validity of QLearningAgent run/execution with reward curve plot
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
        dqn_agent = DqnAgent(env)
        self.assertTrue(isinstance(dqn_agent, DqnAgent), msg=error_msg)
        self.assertIsNotNone(DqnAgent, msg=error_msg)
        self.q_network, all_rewards = dqn_agent.train()
        Helpers.serialObject(dqn_agent, configs.FROZEN_LAKE_DQN_MODEL_FILE_PATH)
        ChartResults.plotRewardCurve(all_rewards, window_size=200)
