from enum import Enum

class Actions(Enum):
    """
    Grid World Actions
    """
    right = 0
    up = 1
    left = 2
    down = 3

class RLAgentType(Enum):
    """
    Enumeration of hedging types
    """
    random = 1
    q_learning = 2
    sarsa = 3
    dqn = 3
    policy_gradient = 4

