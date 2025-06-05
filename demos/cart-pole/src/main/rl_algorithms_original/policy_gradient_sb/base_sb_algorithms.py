import gymnasium as gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    ProgressBarCallback,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    StopTrainingOnNoModelImprovement)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from tqdm import tqdm
from typing import Any, Tuple, List, Dict, Optional
from collections import namedtuple, OrderedDict
import os
from abc import ABC, abstractmethod, abstractproperty

from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.environment.normalize_environment_wrapper import NormalizeEnv

class BasePolicyAlgorithm(ABC):
    """
    Abstract base class for policy gradient algorithms.
    """
    def __init__(
            self,
            env: gym.Env,
            env_name: str,
            hedging_type: HedgingType = HedgingType.gbm,
            total_timesteps: int = int(1e4),
            progress_bar: bool = True,
            is_normalize_obs = False,
            reward_threshold: float = 1.0,
            model_use_case: str = None
    ):
        """
        Constructor
        :param env: Environment
        """
        self.policy_name = "MlpPolicy"
        self.env_name = env_name
        self.rl_problem_title = ""
        self.hedging_type = hedging_type
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar
        self.is_normalize_obs = is_normalize_obs
        self.log_root_path = "./logs"
        self.normalize_obs_stats_path = f"{self.log_root_path}/vec_normalize.pkl"
        self.reward_threshold = reward_threshold
        self.model_use_case: str = "" if model_use_case is None else model_use_case
        self.noise_type = None
        self.noise_std = None

        self.rl_algorithms = {
            RLAgorithmType.ddpg.name: DDPG,
            RLAgorithmType.sac.name: SAC,
            RLAgorithmType.td3.name: TD3,
            RLAgorithmType.ppo.name: PPO
        }

        self.env = env
        self.env_clone = env
        if self.is_normalize_obs:
            self.env = DummyVecEnv([lambda: self.env])
            # Automatically normalize the input features and reward
            self.env = VecNormalize(
                self.env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.
            )
            # normalized_env = NormalizeEnv(self.env)
            # self.env = make_vec_env(lambda: normalized_env, n_envs=1)
        self.model = None

    def createCallbacks(self):
        """
        Creates callbacks for RL agent
        :return:
        """
        if self.is_normalize_obs:
            self.env = VecMonitor(self.env, self.log_path)
        else:
            self.env = Monitor(self.env, self.log_path)

        self.eval_callback_1 = EvalCallback(
            self.env,
            best_model_save_path=self.log_path,
            log_path=self.log_path,
            eval_freq=100,
            deterministic=True,
            render=False
        )

    def createModel(self) -> BaseAlgorithm:
        """
        Creates the RL algorithm/model
        :return: Returns the RL algorithm
        """
        algorithm = self.rl_algorithms[self.algorithm_type_name]
        if self.algorithm_type_name in ["ddpg", "td3"]:
            action_noise = self.computeActionNoise()
            model = algorithm(
                self.policy_name,
                self.env,
                tensorboard_log=self.tensorboard_log_path,
                action_noise=action_noise,
                **self.hyperparameters
            )
        else:
            model = algorithm(
                self.policy_name,
                self.env,
                tensorboard_log=self.tensorboard_log_path,
                **self.hyperparameters
            )
        return model

    def createModelAndTensorboardPaths(self):
        """
        Creates the paths for Model saving and tensorboard paths
        :return: None
        """
        self.log_path = f"./logs/{self.env_name}_{self.algorithm_type_name}_{self.hedging_type}"
        self.tensorboard_log_path = f"./tensorboard/{self.env_name}_{self.algorithm_type_name}/"
        self.model_path = f"./model_{self.algorithm_type_name}/{self.env_name}"
        self.model_root_path = "./model/"
        os.makedirs(self.tensorboard_log_path, exist_ok=True)
        os.makedirs(self.model_root_path, exist_ok=True)

    def train(self):
        """
        Train the PPO RL agent
        """
        self.model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=self.progress_bar,
            callback=[
                self.eval_callback_1,
                # self.save_callback,
                # self.eval_callback_2,
                # self.eval_callback_3,
                # self.monitor_callback,
            ]
        )
        self.model.save(self.model_path)
        if self.is_normalize_obs:
            self.env.save(self.normalize_obs_stats_path)
        del self.model, self.env
        #del self.model

    def evaluate(self, n_test_episodes: int = 50):
        """
        Test the trained RL agent
        """
        algorithm = self.rl_algorithms[self.algorithm_type_name]
        if self.is_normalize_obs:
            vec_env = self._loadNormalizedObsEnvironment()
            model = algorithm.load(self.model_path, env=vec_env)
            # model = algorithm.load(self.model_path)
            mean_reward, std_reward = self._evauateNormalizedEnvironment(
                model=model,
                n_eval_episodes=n_test_episodes,
                deterministic=True
            )
        else:
            self.env = deepcopy(self.env_clone)
            model = algorithm.load(self.model_path, env=self.env_clone)
            mean_reward, std_reward = evaluate_policy(
                model,
                self.env,
                n_eval_episodes=n_test_episodes,
                deterministic=True
            )
        print(f"Mean reward: {mean_reward: .6f} +/- {std_reward: .6f}")

    def _loadNormalizedObsEnvironment(self) -> Any:
        """
        Loads the RL environment with the normalized observation/reward vector
        :return: Returns the normalized RL environment
        :return:
        """
        vec_env = DummyVecEnv([lambda: self.env_clone])
        vec_env = VecNormalize.load(self.normalize_obs_stats_path, vec_env)
        #  do not update them at test time
        vec_env.training = False
        # reward normalization is not needed at test time
        vec_env.norm_reward = False
        return vec_env

    def _evauateNormalizedEnvironment(
        self,
        model: BaseAlgorithm,
        n_eval_episodes: int = 100,
        deterministic: bool = True,
        ) -> Tuple[float, float]:
        """
        Evaluate an RL agent for `num_episodes`.
        :param model: the RL Agent
        :param n_eval_episodes: number of episodes to evaluate it
        :param deterministic: Whether to use deterministic or stochastic actions
        :return: Mean reward for the last `num_episodes`
        """
        # This function will only work for a single environment
        vec_env = model.get_env()
        obs = vec_env.reset()
        all_episode_rewards = []
        for _ in range(n_eval_episodes):
            episode_rewards = []
            done = False
            # Note: SB3 VecEnv resets automatically:
            # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
            # obs = vec_env.reset()
            while not done:
                # _states are only useful when using LSTM policies
                # `deterministic` is to use deterministic actions
                action, _states = model.predict(obs, deterministic=deterministic)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, _info = vec_env.step(action)
                episode_rewards.append(reward)

            all_episode_rewards.append(sum(episode_rewards))

        mean_episode_reward = round(np.mean(all_episode_rewards), 4)
        std_episode_reward = round(np.std(all_episode_rewards), 4)
        return mean_episode_reward, std_episode_reward

    def plotRawMeanRewardCurve(
            self
    ):
        """
        Plot the reward curve results
        """
        results_plotter.plot_results(
            dirs=[self.log_path],
            num_timesteps=self.total_timesteps,
            x_axis=results_plotter.X_TIMESTEPS,
            task_name=self.rl_problem_title + "( raw)"
        )


    def plotSmoothRewardCurve(self):
        """
        Plot a smooth reward curve
        """
        x, y = ts2xy(load_results(self.log_root_path), "timesteps")
        y = self._moving_average(y, window=50)
        if len(y) < len(x):
            # Truncate x
            x = x[len(x) - len(y):]

            fig = plt.figure(self.rl_problem_title)
            plt.plot(x, y)
            plt.xlabel("Number of Timesteps")
            plt.ylabel("Rewards")
            plt.title(self.rl_problem_title + " (Smoothed)")
            plt.show()

    def _moving_average(
            self,
            values: np.ndarray,
            window: int
    ):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")

    def computeActionNoise(self):
        """
        Computes the action noise for TD3 and DDPG RL algorithms
        :return: Action noise
        """
        n_actions = self.env.action_space.shape[-1]
        if self.noise_type == "normal":
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.noise_std * np.ones(n_actions))
        elif self.noise_type == "ornstein-uhlenbeck":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.noise_std * np.ones(n_actions)
            )
        else:
            action_noise = None
        return action_noise

    @property
    @abstractmethod
    def hyperparameters(self) -> Dict[str, Any]:
        """
        Getter of the RL algorithm hyperparameters
        :return: Algorithm parameters
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def algorithm_type_name(self) -> str:
        """
        Getter of the type of RL algorithm name
        :return: RL algorithm type
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def algorithm_type(self) -> RLAgorithmType:
        """
        Getter of the type of RL algorithm
        :return: RL algorithm type
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def preprocessHyperParameters(self) -> Dict[str, Any]:
        """
        Pre-processes the best hyperparameters for the RL algorithm
        :return: Pre-processed hyperparameters
        """
        raise NotImplementedError("Subclasses must implement this method")



