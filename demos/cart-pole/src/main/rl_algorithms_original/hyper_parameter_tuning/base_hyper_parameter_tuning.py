from typing import Any, Dict, Optional
from abc import abstractmethod
import copy
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import TD3, PPO, SAC, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly

from src.main.utility.utils import Helpers
from src.main.environment.env import DynamicHedgingEnv
from src.main.rl_algorithms.hyper_parameter_tuning.trial_evaluation_callback import TrialEvalCallback
from src.main.rl_algorithms.train_evaluate_test.save_on_base_reward_callback import SaveOnBestTrainingRewardCallback
from src.main.utility.enum_types import RLAgorithmType, HedgingType
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
class BaseHyperParameterTuning:
    """
    Abstract base class for SB3 Hyperparameter tuning to support all the
    RL hedger algorithms (TD3, DDPG, SAC and PPO) developed for this project.

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
            env: DynamicHedgingEnv,
            rl_algo_type: RLAgorithmType=RLAgorithmType.td3,
            hedging_type: HedgingType=HedgingType.gbm,
            model_use_case: str = None
    ):
        """
        Constructor
        :param env: Environment
        :param rl_algo_type: RL algorithm type
        :param hedging_type: Hedging type
        :param model_use_case: Model use case description
        """
        self._env = env
        self._n_simulation_paths = env.asset_price_data.shape[0]
        self._n_time_steps_per_episode = env.asset_price_data.shape[1]
        self._rl_algo_type = rl_algo_type
        self._hedging_type = hedging_type
        self._model_use_case = model_use_case
        self._default_hyperparameters = self.setDefaultHyperparameters()
        self._rl_algorithms = {
            RLAgorithmType.ddpg: DDPG,
            RLAgorithmType.sac: SAC,
            RLAgorithmType.td3: TD3,
            RLAgorithmType.ppo: PPO
        }
        self._model = None
        self._best_mean_reward = -np.inf
        self._all_rewards = None
        self._tuned_model_root_path = self.createModelRootPath(
            rl_algo_type=self._rl_algo_type,
            model_use_case=self._model_use_case)
        self._hyperparameter_best_model_path = self.createHyperparameterPath(
                            tuned_model_path=self._tuned_model_root_path
        )
        self._best_model_path = None

    @staticmethod
    def createModelRootPath(rl_algo_type: RLAgorithmType, model_use_case: str) -> str:
        """
        Create the path for model hyperparameter files.
        :param rl_algo_type: RL algorithm type
        :param model_use_case: Model use-case
        :return: model root path
        """
        path = configs2.TUNED_MODEL_PATH.format(rl_algo_type.name, model_use_case)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def createHyperparameterPath(tuned_model_path: str) -> str:
        """
        Create the path for model hyperparameter files.
        :param tuned_model_path: Model root path
        :return: Hyperparameter path
        """
        tuned_model_paraemters_path = f"{tuned_model_path}{configs2.TUNED_PARAMETER_FILE_NAME}"
        return tuned_model_paraemters_path

    @abstractmethod
    def sampleParams(
            self,
            trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Sampler abstract method for RL algorithm hyperparameters.
        :param trial: Optuna Trial
        :return: Sampled parameters
        """
        raise NotImplementedError("Subclasses must implement this method")

    def setDefaultHyperparameters(self) -> Dict[str, Any]:
        """
        Sets the default hyperparameters.
        :return: Default parameters
        """
        match self._rl_algo_type:
            case RLAgorithmType.td3 | RLAgorithmType.ddpg:
                n_actions = self._env.action_space.shape[-1]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                return {
                    "policy": "MlpPolicy",
                    "env": self._env,
                    "action_noise": action_noise,
                    "train_freq": configs2.TRAIN_FREQ,
                }
            case RLAgorithmType.sac:
                return {
                    "policy": "MlpPolicy",
                    "env": self._env,
                }
            case RLAgorithmType.ppo:
                return {
                    "policy": "MlpPolicy",
                    "env": self._env,
                }
            case _:
                raise Exception("Invalid RL algorithm type!!")

    def createModel(self, kwargs: Dict[str, Any]) -> BaseAlgorithm:
        """
        Creates the RL algorithm/model
        :param kwargs: Keyword arguments
        :return: Returns the RL algorithm
        """
        algorithm = self._rl_algorithms[self._rl_algo_type]
        tensorboard_log_path = f"{configs2.HYPER_PARAMETER_TENSORBOARD_FOLDER}/{self._rl_algo_type}_{self._hedging_type}"
        os.makedirs(tensorboard_log_path, exist_ok=True)
        return algorithm(
            **kwargs,
            tensorboard_log=tensorboard_log_path,
        )

    def objective(
            self,
            trial: optuna.Trial
    ) -> float:
        """
        Optimization objective function
        :param trial: Trial
        :return: Returns the Mean reward
        """
        kwargs = self._default_hyperparameters.copy()

        # Sample hyperparameters.
        kwargs.update(self.sampleParams(trial))

        # Create the RL model.
        model = self.createModel(kwargs)

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
        # self._hyperparameter_best_model_path = configs2.HYPER_PARAMETER_BEST_MODEL_PATH.format(self._rl_algo_type.name)
        self._best_model_path = f"{self._tuned_model_root_path}best_model"

        nan_encountered = False
        try:
            model.learn(
                total_timesteps=configs2.N_TUNING_TRAIN_STEPS,
                callback=[eval_callback]
            )
            mean_reward, std_reward = evaluate_policy(
                model, eval_env,
                n_eval_episodes=configs2.N_EVAL_EPISODES,
                deterministic=True
            )
            print(f"Training reward: {mean_reward} +/-{std_reward} for {configs2.N_TUNING_TRAIN_STEPS} steps")

            # Save the model if it has achieved a new best performance.
            if mean_reward > self._best_mean_reward:
                self._best_mean_reward = mean_reward
                model.save(self._best_model_path)
                print(f"New best model saved with mean_reward: {self._best_mean_reward:.2f}")

                self._all_rewards = [x for x in self._env.reward_for_env_episodes]
                print(f"Number of reward values = {len(self._all_rewards)}")


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
        #return mean_reward

    def run(self) -> str:
        """
        Execute the hyperparameter tuning
        :return: The path of the persisted best hyper-parameter tuned model
        """
        # Set pytorch num threads to 1 for faster training.
        torch.set_num_threads(1)

        sampler = TPESampler(n_startup_trials=configs2.N_STARTUP_TRIALS)

        # Do not prune before 1/3 of the max budget is used.
        pruner = MedianPruner(
            n_startup_trials=configs2.N_STARTUP_TRIALS,
            n_warmup_steps=configs2.N_EVALUATIONS)

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize")
        try:
            study.optimize(
                self.objective,
                n_trials=configs2.N_TRIALS,
                timeout=configs2.TUNING_TIMEOUT)
        except KeyboardInterrupt:
            pass
        self._reportResults(study)
        self.plotRewardCurve()
        return self._best_model_path

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
        optimization_history_path = f"{self._tuned_model_root_path}{configs2.HYPER_PARAMETER_HISTORY_PATH}"
        param_importance_path = f"{self._tuned_model_root_path}{configs2.HYPER_PARAMETER_IMPORTANCE_PATH}"
        print(f"Hyper-parameter tuning results will be written to this file: {result_path}")
        print(f"Plot results of the optimization can be found here: {optimization_history_path} "
              f"and {param_importance_path}")
        print(f"The best hyper-parameters computed have been written to {self._hyperparameter_best_model_path}")
        study.trials_dataframe().to_csv(result_path)
        Helpers.serialObject(trial.params, pickle_path=self._hyperparameter_best_model_path)

        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        plotly.offline.plot(fig1, filename=optimization_history_path)
        plotly.offline.plot(fig2, filename=param_importance_path)

    def plotRewardCurve(
            self,
            window: Optional[int] = None,
            end: int = None
    ):
        """
        Plots the reward curve for the best model.
        :param window: window size in number of time-steps
        :param end: end time in number of steps
        :return: None
        """
        reward_curve_path = configs2.HYPER_PARAMETER_REWARD_CURVE_PATH.format(self._rl_algo_type.name)
        reward_curve_data_path = configs2.HYPER_PARAMETER_REWARD_CURVE_DATA_PATH.format(self._rl_algo_type.name)
        if not self._all_rewards:
            print("No reward data available.")
            return

        plt.figure(figsize=(10, 5))
        reward_chunks = Helpers.chunkArray(self._all_rewards, self._n_time_steps_per_episode)
        sum_rewards = [np.sum(x) for x in reward_chunks]
        time_steps = list(range(len(sum_rewards)))

        data_df = pd.DataFrame(
            {
                "time_steps": time_steps,
                "rewards": sum_rewards
            }
        )
        if window:
            if end is None or end > len(time_steps):
                end = time_steps[-1]
            data_df["rewards_sma"] = data_df["rewards"].rolling(window=window).mean()
            start = window + 1
            plt.plot(data_df["time_steps"][start:end], data_df["rewards_sma"][start:end], label="Episode Reward")
        else:
            plt.plot(data_df["time_steps"], data_df["rewards"], label="Episode Reward")

        data_df.to_csv(reward_curve_data_path, index=False)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title("Reward Curve for Best Model (smoothed)")
        plt.legend()
        plt.grid()
        plt.savefig(reward_curve_path)
        plt.close()


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

