import unittest as ut
import inspect
import os
from dataclasses import dataclass, asdict

from src.main.rl_agents.random_agent import RandomAgent
from src.main.configs.random_agent_config import RandomAgentConfig


class RandomAgentTest(ut.TestCase):
    """
    Test suit for RandomAgent
    """
    def test_Random_Agent_Constructor_Is_Valid(self):
        """
        Test the validity of RandomAgent constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        configs = asdict(RandomAgentConfig())
        random_agent = RandomAgent(**configs)
        self.assertTrue(isinstance(random_agent, RandomAgent), msg=error_msg)

    def test_Random_Agent_Run_Is_Valid(self):
        """
        Test the validity of RandomAgent run/execution
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        random_agent_configs = RandomAgentConfig()
        random_agent_configs.render_mode = "human"
        configs = asdict(random_agent_configs)
        random_agent = RandomAgent(**configs)
        self.assertTrue(isinstance(random_agent, RandomAgent), msg=error_msg)
        random_agent.run()

    def test_Random_Agent_Run_With_Reward_Curve_Plot_Is_Valid(self):
        """
        Test the validity of RandomAgent run/execution with reward curve plot
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        random_agent_configs = RandomAgentConfig()
        random_agent_configs.render_mode = "rgb_array"
        configs = asdict(random_agent_configs)
        random_agent = RandomAgent(**configs)
        self.assertTrue(isinstance(random_agent, RandomAgent), msg=error_msg)
        random_agent.run()
        random_agent.plotRewardCurves()


if __name__ == '__main__':
    ut.main()
