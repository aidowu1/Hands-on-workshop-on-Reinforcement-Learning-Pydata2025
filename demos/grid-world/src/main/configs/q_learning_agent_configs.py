import dataclasses

@dataclasses.dataclass
class QLearningAgentConfig:
    """
    Q-Learning Agent configs
    """
    seed: int = 100
    render_mode: str = "rgb_array"
    # render_mode: str = "human"
    size: int = 5
    max_train_episodes: int = 10000
    max_eval_episodes: int = 10
    max_train_steps_per_episode: int = 100
    discount_factor: float = 0.9
    learning_rate: float = 0.1

    # Initial exploration rate
    exploration_rate: float = 1.0

    # Exploration probability at start
    max_exploration_rate: float = 1.0

    # Minimum exploration probability
    min_exploration_rate: float = 0.01

    # Exponential decay rate for exploration prob
    decay_rate: float = 0.005