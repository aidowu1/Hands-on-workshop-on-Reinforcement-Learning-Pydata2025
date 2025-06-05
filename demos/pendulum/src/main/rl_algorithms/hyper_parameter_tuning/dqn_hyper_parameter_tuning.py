from typing import Dict, Any, Optional
import gymnasium as gym
import optuna
import torch.nn as nn

from src.main.utility.enum_types import RLAgorithmType
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.utils import Helpers

class DQNHyperParameterTuning(BaseHyperParameterTuning):
    """
    Class to tune hyperparameters of DQN algorithm.
    Reference: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
    """
    def __init__(
            self,
            env: gym.Env,
            rl_algo_type: RLAgorithmType = RLAgorithmType.ddpg,
            rl_problem_title: str = "CartPole-v0",
    ):
        """
        Constructor
        :param env: RL environment
        :param rl_algo_type: RL algorithm type
        :param rl_problem_title: RL problem title
        """
        super(DQNHyperParameterTuning, self).__init__(env, rl_algo_type, rl_problem_title)

    def sampleParams(
            self,
            trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Sampler for RL algorithm (TD3) hyperparameters.
        :param trial: Optuna Trial
        :return: Sampled parameters
        """
        return self.sampleDDPGParams(trial)

    def sampleDDPGParams(
            self,
            trial: optuna.Trial
    ) -> dict[str, Any]:
        """
        Sampler for DDPG hyperparams (from SB3 Zoo).

        :param trial:
        :return: Hyperparameters for DDPG algorithm
        """
        gamma = trial.suggest_float("gamma", (1.0 - 0.03), (1.0 - 0.0001), log=True)
        # From 2**5=32 to 2**11=2048
        batch_size = trial.suggest_int("batch_size", 2**5, 2**11)

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
        train_freq = trial.suggest_int("train_freq", 1, 10)
        # subsample_steps = trial.suggest_int("subsample_steps", 1, min(train_freq, 8))

        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
        target_update_interval = trial.suggest_int("target_update_interval", 1, 20000, log=True)

        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", 'elu', 'leaky_relu'])

        net_arch = DQNHyperParameterTuning.getNetArchHyperParameter()[net_arch_type]

        activation_fn = DQNHyperParameterTuning.getActivationFunctionHyperParameter()[activation_fn_name]

        hyperparams = {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "train_freq": train_freq,
            # "subsample_steps": subsample_steps,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "target_update_interval": target_update_interval,
            "policy_kwargs": dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
            ),
        }

        return hyperparams

    @staticmethod
    def getNetArchHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
        }

    @staticmethod
    def getActivationFunctionHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}




