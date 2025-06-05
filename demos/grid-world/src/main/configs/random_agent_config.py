import dataclasses

@dataclasses.dataclass
class RandomAgentConfig:
    """
    Random Agent configs
    """
    seed: int = 100
    render_mode: str = "human"
    size: int = 5
    max_episodes: int = 10
    max_steps_per_episode: int = 100
