import numpy as np
from pyvirtualdisplay import Display

from src.main.utility.utils import Helpers
from src.main.configs import global_configs as configs

class EvaluateAgent:
  """
  Evaluate the Q learning RL agent using animation of the simulation runs
  """
  def __init__(
          self,
          agent,
          env,
          n_episodes=3,
          max_steps=100,
          render_mode=configs.RENDER_MODE):
    """
    Constructor
    :param agent: RL Agent
    :param env: Environment
    :param n_episodes: Number of episodes to run
    :param max_steps: Maximum number of steps to run
    :param render_mode: Render mode
    """
    self.agent = agent
    self.env = env
    self.render_mode = render_mode
    self.n_episodes = n_episodes
    self.max_steps = max_steps
    if self.render_mode == "human":
        self.display = None
    else:
        self.display = Display(visible=0, size=(400, 400))
        self.display.start()
    self.states = []

    self.images = []

  def _evaluate(self):
    """
    Evaluate the agent
    """
    for ep in range(self.n_episodes):
      s, _ = self.env.reset()
      self.states.append(s)

      # Rollout under greedy policy
      for _ in range(self.max_steps):
          a = np.argmax(self.agent.Q[s])
          s, _, done, _, _ = self.env.step(a)
          if self.render_mode == "human":
            self.env.render()
          else:
            self.images.append(self.env.render())
          self.states.append(s)
          if done:
              break

      # self.env.close()

  def run(self):
    """
    Run the RL evaluation with animation
    """
    self._evaluate()
    if self.render_mode != "human":
        Helpers.animateEnvironment(self.images)
