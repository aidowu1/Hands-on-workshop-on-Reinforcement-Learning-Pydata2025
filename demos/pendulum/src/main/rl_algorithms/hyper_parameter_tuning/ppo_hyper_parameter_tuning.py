from typing import Dict, Any
import gymnasium as gym
import optuna
from rl_zoo3 import linear_schedule
import torch.nn as nn

from src.main.utility.enum_types import RLAgorithmType
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning

class PPOHyperParameterTuning(BaseHyperParameterTuning):
    """
    Class to tune hyperparameters of PPO algorithm.
    Reference: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """
    def __init__(
            self,
            env: gym.Env,
            rl_algo_type: RLAgorithmType = RLAgorithmType.ppo,
            rl_problem_title: str = "Pendulum-v1",
    ):
        """
        onstructor
        :param env: RL environment
        :param rl_algo_type: RL algorithm type
        :param rl_problem_title: RL problem title
        """
        super(PPOHyperParameterTuning, self).__init__(env, rl_algo_type, rl_problem_title)

    def sampleParams(
            self,
            trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Sampler for RL algorithm (TD3) hyperparameters.
        :param trial: Optuna Trial
        :return: Sampled parameters
        """
        return self.samplePPOParams(trial)

    def samplePPOParams(
            self,
            trial: optuna.Trial
    ) -> dict[str, Any]:
        """
        Sampler for PPO hyperparams (from SB3 Zoo).

        :param trial:
        :return: Hyperparameters for PPO algorithm
        """
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float("vf_coef", 0, 1)
        net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

        # Uncomment for gSDE (continuous actions)
        log_std_init = trial.suggest_float("log_std_init", -4, 1)

        # Uncomment for gSDE (continuous action)
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])

        # Orthogonal initialization
        # ortho_init = False
        ortho_init = trial.suggest_categorical('ortho_init', [False, True])

        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", 'elu', 'leaky_relu'])

        # lr_schedule = "constant"
        # Uncomment to enable learning rate schedule
        lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        if lr_schedule == "linear":
            learning_rate = linear_schedule(learning_rate)

        # TODO: account when using multiple envs
        if batch_size > n_steps:
            batch_size = n_steps

        # Independent networks usually work best
        # when not working with images
        net_arch = PPOHyperParameterTuning.getNetArchHyperParameter()[net_arch_type]

        activation_fn = PPOHyperParameterTuning.getActivationFunctionHyperParameter()[activation_fn_name]

        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            "sde_sample_freq": sde_sample_freq,
            "policy_kwargs": dict(
                log_std_init=log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        }
    @staticmethod
    def getNetArchHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {
            "tiny": dict(pi=[64], vf=[64]),
            "small": dict(pi=[64, 64], vf=[64, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256]),
        }

    @staticmethod
    def getActivationFunctionHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
