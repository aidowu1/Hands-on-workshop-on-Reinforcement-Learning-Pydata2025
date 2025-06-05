import unittest as ut
import inspect
import os
from pprint import pprint

from src.main.evaluations.evaluate_rl_agent import QLearningAgent, EvaluateRLAgent
from src.main.utility.utils import Helpers
import src.main.configs.global_configs as configs



class EvaluateRLAgentTest(ut.TestCase):
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

    def test_EvaluateRLAgent_Agent_Constructor_Is_Valid(self):
        """
        Test the validity of EvaluateRLAgent constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = Helpers.deserializeObject(configs.GRIDWORLD_QL_MODEL_FILE_PATH)
        self.assertIsNotNone(agent, msg=error_msg)
        rl_evaluator = EvaluateRLAgent(agent=agent)
        print("q_table:")
        pprint(agent.q_table)
        self.assertIsNotNone(rl_evaluator, msg=error_msg)

    def test_EvaluateRLAgent_Agent_Evaluation_Is_Valid(self):
        """
        Test the validity of EvaluateRLAgent evaluation function
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        agent = Helpers.deserializeObject(configs.GRIDWORLD_QL_MODEL_FILE_PATH)
        self.assertIsNotNone(agent, msg=error_msg)
        rl_evaluator = EvaluateRLAgent(agent=agent)
        self.assertIsNotNone(rl_evaluator, msg=error_msg)
        rl_evaluator.evaluate()

