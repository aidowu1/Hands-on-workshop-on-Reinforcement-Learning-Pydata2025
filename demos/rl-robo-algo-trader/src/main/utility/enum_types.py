from enum import Enum

class RLAgentType(Enum):
    """
    Enumeration of RL Agent types
    """
    random = 1
    q_learning = 2
    sarsa = 3
    dqn = 3
    policy_gradient = 4

class RLAgorithmType(Enum):
    """
    Enumeration of RL algorithm types
    """
    ddpg = 1
    td3 = 2
    sac = 3
    ppo = 4
    dqn = 5

