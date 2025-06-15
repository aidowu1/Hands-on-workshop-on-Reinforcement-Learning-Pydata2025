import unittest as ut
import inspect
import os
from dataclasses import asdict

from src.main.rl_agents.rl_algo_trading_agent import RLAlgoTradingAgent
from src.main.utility.enum_types import RLAgorithmType
import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.utility.chart_results import ChartResults


class RLAlgoTradingAgentTest(ut.TestCase):
    """
    Test suit for RLAlgoTradingAgent
    """
    def setUp(self):
        """
        Setup test environment
        :return:
        """
        self.current_path = Helpers.getPojectRootPath()
        print(f"Current path is: {self.current_path}...{configs.NEW_LINE}")
        os.chdir(self.current_path)

    def test_RLAlgoTradingAgent_DQN_Agent_Constructor_Is_Valid(self):
        """
        Test the validity of RLAlgoTradingAgent constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rl_algo = RLAgorithmType.dqn
        agent = RLAlgoTradingAgent(rl_algorithm_type=rl_algo)
        self.assertIsNotNone(agent, msg=error_msg)

    def test_RLAlgoTradingAgent_Random_Agent_Training_Is_Valid(self):
        """
        Test the validity of RLAlgoTradingAgent random agent
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rl_algo = RLAgorithmType.dqn
        agent = RLAlgoTradingAgent(rl_algorithm_type=rl_algo)
        self.assertIsNotNone(agent, msg=error_msg)
        infos = agent.trainRLWithRandomAgent(n_episodes=100)
        self.assertIsNotNone(infos, msg=error_msg)

    def test_RLAlgoTradingAgent_DQN_Agent_Training_Is_Valid(self):
        """
        Test the validity of RLAlgoTradingAgent training
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rl_algo = RLAgorithmType.dqn
        agent = RLAlgoTradingAgent(rl_algorithm_type=rl_algo)
        self.assertIsNotNone(agent, msg=error_msg)
        infos = agent.train()
        self.assertIsNotNone(infos, msg=error_msg)

    def test_RLAlgoTradingAgent_DQN_Agent_Validation_Is_Valid(self):
        """
        Test the validity of RLAlgoTradingAgent validation
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rl_algo = RLAgorithmType.dqn
        agent = RLAlgoTradingAgent(rl_algorithm_type=rl_algo)
        self.assertIsNotNone(agent, msg=error_msg)
        rewards, infos = agent.validate()
        self.assertIsNotNone(infos, msg=error_msg)

    def test_RLAlgoTradingAgent_PPO_Agent_Training_And_Validation_Is_Valid(self):
        """
        Test the validity of RLAlgoTradingAgent PPO training & validation
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rl_algo = RLAgorithmType.ppo
        agent = RLAlgoTradingAgent(rl_algorithm_type=rl_algo)
        self.assertIsNotNone(agent, msg=error_msg)
        infos_all_episode = agent.train()
        infos_last_episode = infos_all_episode[-1]
        self.assertIsNotNone(infos_last_episode, msg=error_msg)
        agent.plotTrainRewardCurve()
        agent.reportAgentBehaviour(agent.train_env, infos_last_episode)
        rewards, infos = agent.validate()
        self.assertIsNotNone(infos, msg=error_msg)

    def test_RLAlgoTradingAgent_DQN_Agent_Training_And_Validation_Is_Valid(self):
        """
        Test the validity of RLAlgoTradingAgent DQN training & validation
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rl_algo = RLAgorithmType.dqn
        agent = RLAlgoTradingAgent(rl_algorithm_type=rl_algo)
        self.assertIsNotNone(agent, msg=error_msg)
        infos_all_episode = agent.train()
        infos_last_episode = infos_all_episode[-1]
        self.assertIsNotNone(infos_last_episode, msg=error_msg)
        agent.reportAgentBehaviour(agent.train_env, infos_last_episode)
        rewards, infos = agent.validate()
        self.assertIsNotNone(infos, msg=error_msg)

