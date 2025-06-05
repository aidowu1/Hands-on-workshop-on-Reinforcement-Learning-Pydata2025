from typing import Any, Dict
import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import numpy as np
import plotly

from src.main.rl_algorithms.hyper_parameter_tuning.trial_evaluation_callback import TrialEvalCallback
from src.main.utility.enum_types import RLAgorithmType
import src.main.configs_rl as configs2

class TD3HyperParameterTuning:
    """
    SB3 Hyperparameter tuning for TD3 RL hedger algorithm
    This implementation is inspired by the  Antonin’s Raffin (from Stable-baselines)  ICRA 2022
    presentation titled: Automatic Hyperparameter Optimization
    located here: https://araffin.github.io/slides/icra22-hyperparam-opt/

    A summary of the steps for implementing the SB3 Hyperparameter Tuning include:
        - Step 1: Define the sample parameters for the Optuna optimization
        - Step 2: Specification Trial evaluation callback class
        - Step 3: Specify the objective function of the hyperparameter optimization routine
        - Step 4: Run the hyperparameter routine
    """
    def __init__(
            self,
            env: gymnasium.Env,
            rl_algo_type: RLAgorithmType=RLAgorithmType.td3
    ):
        """
        Constructor
        """
        self._env = env
        self._rl_algo_type = rl_algo_type
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        self.default_hyperparameters = {
            "policy": "MlpPolicy",
            "env": env,
            "action_noise": action_noise,
            "train_freq": configs2.TRAIN_FREQ,
        }

    def sampleParams(
            self,
            trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Sampler for RL algorithm (TD3) hyperparameters.
        :param trial: Optuna Trial
        :return: Sampled parameters
        """
        gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 512)
        tau = trial.suggest_float("tau", 0.0001, 0.3, log=True)
        learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
        target_policy_noise = trial.suggest_float("target_policy_noise", 1e-5, 1, log=True)
        net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

        # Display true values.
        trial.set_user_attr("gamma_", gamma)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("tau", tau)
        trial.set_user_attr("learning_rate", learning_rate)

        # net_arch = [
        #     {"pi": [32, 32], "qf": [32, 32]} if net_arch == "tiny" else {"pi": [64, 64], "qf": [64, 64]}
        # ]
        if net_arch == "tiny":
            net_arch_config = dict(pi=[32, 32], qf=[32, 32])
        else:
            net_arch_config = dict(pi=[64, 64], qf=[64, 64])

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "learning_rate": learning_rate,
            "target_policy_noise": target_policy_noise,
            "policy_kwargs": {
                "net_arch": net_arch_config,
                "activation_fn": activation_fn,
            },
        }

    def objective(
            self,
            trial: optuna.Trial
    ) -> float:
        """
        Optimization objective function
        :param trial: Trial
        :return: Returns the Mean reward
        """
        kwargs = self.default_hyperparameters.copy()
        # Sample hyperparameters.
        kwargs.update(self.sampleParams(trial))
        # Create the RL model.
        model = TD3(**kwargs)
        # Create env used for evaluation.
        eval_env = Monitor(self._env)
        # Create the callback that will periodically evaluate and report the performance.
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=configs2.N_EVAL_EPISODES,
            eval_freq=configs2.EVAL_FREQ,
            deterministic=True
        )

        nan_encountered = False
        try:
            model.learn(
                configs2.N_TIMESTEPS,
                callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)
            nan_encountered = True
        finally:
            # Free memory.
            model.env.close()
            eval_env.close()

        # Tell the optimizer that the trial failed.
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

    def run(self):
        """
        Execute the hyperparameter tuning
        :return: None
        """
        # Set pytorch num threads to 1 for faster training.
        torch.set_num_threads(1)

        sampler = TPESampler(n_startup_trials=configs2.N_STARTUP_TRIALS)

        # Do not prune before 1/3 of the max budget is used.
        pruner = MedianPruner(
            n_startup_trials=configs2.N_STARTUP_TRIALS,
            n_warmup_steps=configs2.N_EVALUATIONS // 3)

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize")
        try:
            study.optimize(
                self.objective,
                n_trials=configs2.N_TRIALS,
                timeout=600)
        except KeyboardInterrupt:
            pass
        self._reportResults(study)



    def _reportResults(
            self,
            study: optuna.Study):
        """
        Report hyperparameter optimization results
        :param trial: Trial
        :return: None
        """
        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))

        # Write report
        result_path = configs2.HYPER_PARAMETER_RESULT_PATH.format(self._rl_algo_type.name)
        optimization_history_path = configs2.HYPER_PARAMETER_HISTORY_PATH.format(self._rl_algo_type.name)
        param_importance_path = configs2.HYPER_PARAMETER_IMPORTANCE_PATH.format(self._rl_algo_type.name)
        print(f"Hyper-parameter tuning results will be written to this file: {result_path}")
        print(f"Plot results of the optimization can be found here: {optimization_history_path} "
              f"and {param_importance_path}")
        study.trials_dataframe().to_csv(result_path)

        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        plotly.offline.plot(fig1, filename=optimization_history_path)
        plotly.offline.plot(fig2, filename=param_importance_path)

        fig1.show()
        fig2.show()
