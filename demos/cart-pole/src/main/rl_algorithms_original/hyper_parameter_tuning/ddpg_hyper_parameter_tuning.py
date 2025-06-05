import copy
from typing import Dict, Any, Optional
import numpy as np
import gymnasium
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch.nn as nn

from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.environment.env import DynamicHedgingEnv
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning

class DDPGHyperParameterTuning(BaseHyperParameterTuning):
    """
    Class to tune hyperparameters of DDPG algorithm.
    Reference: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """
    def __init__(
            self,
            env: DynamicHedgingEnv,
            rl_algo_type: RLAgorithmType = RLAgorithmType.ddpg,
            hedging_type: HedgingType=HedgingType.gbm,
            model_use_case: str = None
    ):
        """
        Constructor
        :param env: RL environment
        :param rl_algo_type: RL algorithm type
        :param hedging_type: Hedging type
        :param model_use_case: Model use case description
        """
        super(DDPGHyperParameterTuning, self).__init__(env, rl_algo_type, hedging_type, model_use_case)

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
        n_actions = self._env.action_space.shape[-1]
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        # Polyak coeff
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
        gradient_steps = train_freq

        noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
        noise_std = trial.suggest_float("noise_std", 0, 1)

        # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", 'elu', 'leaky_relu'])

        net_arch = DDPGHyperParameterTuning.getNetArchHyperParameter()[net_arch_type]

        activation_fn = DDPGHyperParameterTuning.getActivationFunctionHyperParameter()[activation_fn_name]

        hyperparams = {
            "gamma": gamma,
            "tau": tau,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "policy_kwargs": dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
            ),
        }

        hyperparams = DDPGHyperParameterTuning.setNoiseHyperParameter(
            hyperparams, n_actions, noise_type, noise_std
        )
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

    @staticmethod
    def setNoiseHyperParameter(
            hyperparams: Dict[str, Any],
            n_actions: int,
            noise_type: Optional[str],
            noise_std: float
    ) -> Dict[str, Any]:
        """
        Sets the noise hyperparameters.
        :param hyperparams: Dictionary of hyperparameters
        :param n_actions: Number of actions
        :param noise_type: Type of noise
        :param noise_std: Noise standard deviation value
        :return: Modified hyperparameters
        """
        hyperparams_new = copy.deepcopy(hyperparams)
        if noise_type:
            if noise_type == "normal":
                hyperparams_new["action_noise"] = NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=noise_std * np.ones(n_actions))
            elif noise_type == "ornstein-uhlenbeck":
                hyperparams_new["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=noise_std * np.ones(n_actions)
                )
        return hyperparams_new


