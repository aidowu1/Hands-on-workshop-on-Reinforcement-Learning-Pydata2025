from dataclasses import dataclass

from src.main.configs.q_learning_agent_configs import QLearningAgentConfig

@dataclass
class SarsaAgentConfig(QLearningAgentConfig):
    super().__init__()