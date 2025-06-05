import dataclasses
import gymnasium as gym

from src.main.configs import global_configs as configs

@dataclasses.dataclass
class QLearningAgentConfig:
    """
    Q-Learning Agent configs
    """
    # Environment
    env: gym.Env = configs.FROZEN_LAKE_ENV
    seed: int = configs.SEED
    alpha:float = configs.Q_LEARN_ALPHA
    gamma: float = configs.Q_LEARN_GAMMA
    epsilon: float = configs.Q_LEARN_EPSILON
    epsilon_decay: float = configs.Q_LEARN_EPSILON_DECAY
    min_epsilon: float = configs.Q_LEARN_MIN_EPSILON
    n_episodes: int = configs.Q_LEARN_N_EPISODES
    max_steps: int = configs.Q_LEARN_MAX_STEPS

