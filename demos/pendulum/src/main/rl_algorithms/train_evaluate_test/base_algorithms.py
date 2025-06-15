import os
import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, DDPG, SAC, PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
import os
from abc import ABC, abstractmethod

import src.main.configs.global_configs as configs
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType
from src.main.utility.logging import Logger


class BaseRLAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    """
    def __init__(
            self,
            env: gym,
            rl_algorithm_type: RLAgorithmType,
            rl_problem_title: str ="Pendulum-v1",
            n_train_episodes: int = configs.SB3_N_EPISODES,
            n_eval_episodes: int = configs.SB3_N_EVALUATION_EPISODES,
            max_steps: int = configs.SB3_MAX_STEPS,
            check_freq: int = configs.SB3_CHECK_FREQUENCY,
            reward_threshold: float = configs.SB3_REWARD_THRESHOLD,
            max_train_time_steps: int = configs.SB3_MAX_TRAIN_TIME_STEPS


    ):
        """
        Constructor
        :param env: Environment
        """
        self._logger = Logger().getLogger()
        self._policy_name = "MlpPolicy"
        self._rl_problem_title = rl_problem_title
        self._n_train_episodes = n_train_episodes
        self._n_eval_episodes = n_eval_episodes
        self._n_steps = max_steps
        self._max_train_time_steps = max_train_time_steps
        self._check_freq = check_freq
        self._reward_threshold = reward_threshold
        self._rl_algo_type = rl_algorithm_type
        self._log_dir = self.createLogPath()
        self._tuned_model_root_path = BaseHyperParameterTuning.createModelRootPath(
            rl_algo_type=self._rl_algo_type,
            problem_title=self._rl_problem_title)
        self._model_path = self.createSaveModelPath()
        self._plot_dir = self.createPlotPath()
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._plot_dir, exist_ok=True)
        self._tensorboard_dir = f"tensorboard/{self._rl_problem_title}_{self._rl_algo_type.name}"
        self._env = Monitor(env, self._log_dir)
        self._seed = configs.SEED

        # Specify supported RL algorithms
        self._rl_algorithms = {
            RLAgorithmType.ddpg: DDPG,
            RLAgorithmType.sac: SAC,
            RLAgorithmType.td3: TD3,
            RLAgorithmType.ppo: PPO,
            RLAgorithmType.dqn: DQN
        }
        self._model = None
        self._rewards = []


    @abstractmethod
    def train(self):
        """
        Trains the RL algorithm.
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def tuned_hyper_parameters(self) -> Dict[str, Any]:
        """
        Gets and pre-processes the best tuned hyperparameters for the RL algorithm
        :return: Best hyperparameters
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def non_tuned_hyperparameters(self) -> Dict[str, Any]:
        """
        Getter for the non-turned hyperparameters
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def hyper_parameters(self) -> Dict[str, Any]:
        """
        Getter for RL algorithm hyperparameters
        :return: RL algorithm hyperparameters
        """
        if configs.SB3_IS_USE_HYPER_PARAMETER_TUNING:
            self._logger.info("Training is using tuned hyperparameters...")
            self._logger.info(f"Hyperparameters are:\n{self.tuned_hyper_parameters}")
            return self.tuned_hyper_parameters
        else:
            self._logger.info("Training is using non-tuned hyperparameters...")
            self._logger.info(f"Hyperparameters are:\n{self.non_tuned_hyperparameters}")
            return self.non_tuned_hyperparameters

    def evaluate(
            self,
            model_path: str = None,
            env: gym.Env = gym.make("Pendulum-v1", render_mode="human")
    ) -> List[float]:
        """
        Evaluates the RL algorithm.
        :param model_path: Model path
        :param selected_path_index: Selected simulation path index
        :return: Evaluation results
        """
        if model_path is None:
            model_path = self._model_path
        algorithm = self._rl_algorithms[self._rl_algo_type]
        self._model = algorithm.load(model_path, env=self._env)

        for _ in range(self._n_eval_episodes ):
            states, info = env.reset(seed=self._seed)
            total_reward = 0
            for _ in range(self._n_steps):
                action_array, _states = self._model.predict(states)
                action = np.array([action_array.item()])
                states, reward, done, terminated, info = env.step(action)
                total_reward += reward
                self._rewards.append(total_reward)

                env.render()

                if done or terminated:
                    break

        return self._rewards

    def createModel(self) -> BaseAlgorithm:
        """
        Creates the RL algorithm/model
        :return: Returns the RL algorithm
        """
        algorithm = self._rl_algorithms[self._rl_algo_type]
        match self._rl_algo_type:
            case RLAgorithmType.td3 | RLAgorithmType.ddpg:
                return algorithm(
                            self._policy_name,
                            self._env,
                            tensorboard_log=self._tensorboard_dir,
                            **self.hyper_parameters
                        )
            case RLAgorithmType.sac | RLAgorithmType.ppo | RLAgorithmType.dqn:
                return algorithm(
                    self._policy_name,
                    self._env,
                    tensorboard_log=self._tensorboard_dir,
                    **self.hyper_parameters
                )
            case _:
                raise Exception("Invalid RL algorithm type!!")

    def createLogPath(self) -> str:
        """
        Creates the log path
        :return: Log path
        """
        log_path = f"./logs/{self._rl_problem_title}_{self._rl_algo_type.name}"
        return log_path

    def createPlotPath(self) -> str:
        """
        Creates the plot path
        :return: Plot path
        """
        plot_path = f"./Plots/{self._rl_problem_title}_{self._rl_algo_type.name}"
        return plot_path

    def createSaveModelPath(
            self,
            custom_model_path: str = None
    ) -> str:
        """
        Creates the save model path
        :return: Model save path
        """
        if custom_model_path is None:
            model_path = f"{self._tuned_model_root_path}best_model"
        else:
            model_path = f"{custom_model_path}/best_model"
        return model_path

    @property
    def trained_model(self) -> BaseAlgorithm:
        """
        Getter for the trained model
        :return: Trained model
        """
        return self._model

    @property
    def algorithm(self) -> Any:
        """
        Getter for the trained model
        :return: Trained model
        """
        return self._rl_algorithms[self._rl_algo_type]

    @property
    def problem_title(self) -> str:
        """
        Getter for the problem title
        :return: Problem title
        """
        return self._rl_problem_title

    @property
    def algorithim_type(self) -> RLAgorithmType:
        """
        Getter for the algorithm type
        :return: Algorithm type
        """
        return self._rl_algo_type

