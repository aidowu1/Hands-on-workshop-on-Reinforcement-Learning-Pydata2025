from typing import Dict, Any
import gymnasium
import optuna
import torch.nn as nn

from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.environment.env import DynamicHedgingEnv
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning

class SACHyperParameterTuning(BaseHyperParameterTuning):
    """
    Class to tune hyperparameters of SAC algorithm.
    Reference: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """
    def __init__(
            self,
            env: DynamicHedgingEnv,
            rl_algo_type: RLAgorithmType = RLAgorithmType.sac,
            hedging_type: HedgingType = HedgingType.gbm,
            model_use_case: str = None
    ):
        """
        Constructor
        :param env: RL environment
        :param rl_algo_type: RL algorithm type
        """
        super(SACHyperParameterTuning, self).__init__(env, rl_algo_type, hedging_type, model_use_case)

    def sampleParams(
            self,
            trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Sampler for RL algorithm (TD3) hyperparameters.
        :param trial: Optuna Trial
        :return: Sampled parameters
        """
        return self.sampleSACParams(trial)

    def sampleSACParams(
            self,
            trial: optuna.Trial
    ) -> dict[str, Any]:
        """
        Sampler for SAC hyperparams.

        :param trial:
        :return:
        """
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        learning_starts = trial.suggest_categorical("learning_starts", [1000, 10000, 20000])
        # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
        # Polyak coeff
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
        # gradient_steps takes too much time
        # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
        gradient_steps = train_freq
        # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
        ent_coef = "auto"
        # You can comment that out when not using gSDE
        log_std_init = trial.suggest_float("log_std_init", -4, 1)
        # NOTE: Add "verybig" to net_arch when tuning HER
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
        activation_fn_name = trial.suggest_categorical('activation_fn', ["tanh", "relu", 'elu', 'leaky_relu'])

        net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
        }[net_arch_type]

        activation_fn = SACHyperParameterTuning.getActivationFunctionHyperParameter()[activation_fn_name]

        target_entropy = "auto"
        # if ent_coef == 'auto':
        #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
        #     target_entropy = trial.suggest_float('target_entropy', -10, 10)

        hyperparams = {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef,
            "tau": tau,
            "target_entropy": target_entropy,
            "policy_kwargs": dict(
                log_std_init=log_std_init,
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
            "small": [64, 64, 64],
            "medium": [128, 128, 128],
            "big": [256, 256, 256],
        }

    @staticmethod
    def getActivationFunctionHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}

