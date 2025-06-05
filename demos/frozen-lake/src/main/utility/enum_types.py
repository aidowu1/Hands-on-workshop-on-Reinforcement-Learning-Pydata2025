from enum import Enum

class RLAgentType(Enum):
    """
    Enumeration of hedging types
    """
    random = 1
    q_learning = 2
    sarsa = 3
    dqn = 3
    policy_gradient = 4

