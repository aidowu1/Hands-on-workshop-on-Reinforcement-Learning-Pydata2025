import random
import numpy as np                                       # numerical ops
import gymnasium as gym                                  # Gymnasium environments
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pyvirtualdisplay import Display                     # headless display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
from typing import List, Any
import os, sys

from src.main.utility.utils import Helpers
from src.main.utility.replay_buffers import ReplayBuffer
from src.main.utility.q_network import QNetwork

class DqnAgent:
  """
  DQN agent training loop
  """
  def __init__(
      self,
      env,
      n_episodes=1200,
      batch_size=64,
      gamma=0.99,
      lr=1e-3,
      target_update=50,
      epsilon = 1.0,
      eps_decay = 0.995,
      eps_min = 0.01,
      eval_episodes = 2,
      eval_max_steps = 100,
  ):
    """
    Initializes the DQN agent training loop
    :param env: Environment
    :param n_episodes: Number of episodes
    :param batch_size: Batch size
    :param gamma: Discount factor
    :param lr: Learning rate
    :param target_update: Target update frequency
    :param epsilon: Epsilon
    :param eps_decay: Epsilon decay
    :param eps_min: Epsilon minimum
    """
    self.env = env
    self.n_episodes = n_episodes
    self.batch_size = batch_size
    self.gamma = gamma
    self.lr = lr
    self.target_update = target_update
    self.epsilon = epsilon
    self.eps_decay = eps_decay
    self.eps_min = eps_min

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.state_size = env.observation_space.n
    self.action_size = env.action_space.n

    self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
    self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
    self.target_network.load_state_dict(self.q_network.state_dict())
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
    self.replay_buffer = ReplayBuffer()
    self.states = []
    self.images = []
    self.eval_episodes = eval_episodes
    self.eval_max_steps = eval_max_steps

  def selectAction(
    self,
    state: Any) -> int:
    """
    Selects an action
    :param state: State
    :return: Action
    """
    if random.random() < self.epsilon:
        return random.randrange(self.action_size)
    with torch.no_grad():
        return self.q_network(torch.FloatTensor(state)).argmax().item()

  def train(self):
    """
    Trains the DQN agent
    """
    all_rewards = []

    for ep in range(1, self.n_episodes+1):
        state, _ = self.env.reset()
        state = Helpers.toOneHotEncoding(state, self.state_size)
        total_r = 0

        done = False
        while not done:
            action = self.selectAction(state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = Helpers.toOneHotEncoding(next_state, self.state_size)

            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_r += reward

            if len(self.replay_buffer) >= self.batch_size:
                s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)
                s  = torch.FloatTensor(s)
                a  = torch.LongTensor(a)
                r  = torch.FloatTensor(r)
                s2 = torch.FloatTensor(s2)
                d  = torch.BoolTensor(d)

                # Current Q
                q_vals = self.q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)
                # Next Q from target network
                next_q = self.target_network(s2).max(1)[0]
                next_q[d] = 0.0  # zero for terminal states
                # DQN loss
                target = r + self.gamma * next_q
                loss = nn.functional.mse_loss(q_vals, target.detach())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        all_rewards.append(total_r)

        # Periodic target network update
        if ep % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if ep % 200 == 0:
            print(f"Episode {ep}, AvgReward {np.mean(all_rewards[-200:]):.3f}")

    return self.q_network, all_rewards

  def evaluate(
      self,
      env: gym.Env,
      render_mode: str="human"):
    """
    Evaluate the agent
    """
    for ep in range(self.eval_episodes):
      s, _ = env.reset()
      state = Helpers.toOneHotEncoding(s, self.state_size)
      self.states.append(state)

      # Rollout under greedy policy
      for _ in range(self.eval_max_steps):
          # a = np.argmax(self.agent.Q[s])
          # s, _, done, _, _ = self.env.step(a)
          action = self.q_network(torch.FloatTensor(state)).argmax().item()
          state, reward, done, _, _ = env.step(action)
          state = Helpers.toOneHotEncoding(state, self.state_size)

          if render_mode == "human":
            env.render()
          else:
            self.images.append(env.render())
          self.states.append(state)
          if done:
              break

      # self.env.close()

  @property
  def q_net(self) -> QNetwork:
    """
    Getter for the Q network
    :return: Q network
    """
    return self.q_network