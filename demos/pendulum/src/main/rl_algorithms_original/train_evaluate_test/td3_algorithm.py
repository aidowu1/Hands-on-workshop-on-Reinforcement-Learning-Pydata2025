import gymnasium as gym
from typing import Dict, Any
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.rl_algorithms.train_evaluate_test.save_on_base_reward_callback import SaveOnBestTrainingRewardCallback
from src.main.rl_algorithms.train_evaluate_test.base_algorithms import BaseRLAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning import TD3HyperParameterTuning
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers

class TD3TrainAlgorithm(BaseRLAlgorithm):
    """
    TD3 Algorithm
    """

    def __init__(
            self,
            env: gym.Env,
            rl_problem_title: str = "RL Delta Hedger",
            hedging_type: HedgingType = HedgingType.gbm,
            max_steps: int = configs2.N_TUNING_TRAIN_STEPS,
            check_freq: int = configs2.CHECKPOINT_FREQ,
            model_use_case: str = None
    ):
        """
        
        :param env: RL environment
        :param rl_algorithm_type:
        :param rl_problem_title: 
        :param max_steps: 
        """
        super(TD3TrainAlgorithm, self).__init__(
            env,
            RLAgorithmType.td3,
            hedging_type,
            rl_problem_title,
            max_steps,
            check_freq,
            model_use_case
        )
        self._logger.info(f"Start of Reinforcement learning for environment: {self._rl_problem_title.upper()}")
        self._logger.info(f"This RL environment uses a {self._rl_algo_type.name.upper()} RL algorithm agent")
        self._model = self.createModel()

    def train(self):
        """
        Trains the RL algorithm.
        :param check_freq: Check frequency
        :return:
        """
        self._logger.info(f"Start of RL algorithm training, this might take a while..")
        callback = SaveOnBestTrainingRewardCallback(check_freq=self._check_freq, log_dir=self._log_dir)
        self._model.learn(total_timesteps=self._max_train_time_steps, callback=callback)
        self._model.save(path=self._model_path)
        self._logger.info(f"End of RL algorithm training")

    @property
    def tuned_hyper_parameters(self) -> Dict[str, Any]:
        """
        Gets and pre-processes the best tuned hyperparameters for the RL algorithm
        :return: Best hyperparameters
        """
        #best_hyper_parameters_path = configs2.HYPER_PARAMETER_BEST_VALUES.format(self._rl_algo_type.name)
        best_hyper_parameters_path = BaseHyperParameterTuning.createHyperparameterPath(self._tuned_model_root_path)
        best_hyper_parameters_all = Helpers.deserializeObject(best_hyper_parameters_path)

        best_net_architecture = TD3HyperParameterTuning.getNetArchHyperParameter()
        best_activation_function = TD3HyperParameterTuning.getActivationFunctionHyperParameter()
        net_arch = best_net_architecture[best_hyper_parameters_all[configs2.HYPER_PARAMETER_NET_ARCH]]
        activation_fn = best_activation_function[best_hyper_parameters_all[configs2.HYPER_PARAMETER_ACTIVATION_FN]]
        noise_type = best_hyper_parameters_all.get(configs2.HYPER_PARAMETER_NOISE_TYPE)
        noise_std = best_hyper_parameters_all.get(configs2.HYPER_PARAMETER_NOISE_STD)
        n_actions = self._env.action_space.shape[-1]
        best_hyper_parameters_all = BaseHyperParameterTuning.setNoiseHyperParameter(
            best_hyper_parameters_all, n_actions, noise_type, noise_std
        )

        filter_list = [configs2.HYPER_PARAMETER_NOISE_TYPE, configs2.HYPER_PARAMETER_NOISE_STD,
                       configs2.HYPER_PARAMETER_NET_ARCH, configs2.HYPER_PARAMETER_ACTIVATION_FN]
        best_hyper_parameters = Helpers.filterDict(best_hyper_parameters_all, filter_list)
        policy_kwargs = {
            configs2.HYPER_PARAMETER_NET_ARCH: net_arch,
            configs2.HYPER_PARAMETER_ACTIVATION_FN: activation_fn,
        }
        best_hyper_parameters[configs2.HYPER_PARAMETER_POLICY_KWARGS] = policy_kwargs
        return best_hyper_parameters

    @property
    def non_tuned_hyperparameters(self) -> Dict[str, Any]:
        """
        Getter for the non-turned hyperparameters
        :return:
        """
        n_actions = self._env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        return {
            # "action_noise": action_noise,
            "gamma": 0.9999,
            "learning_rate": 0.0005,
            "batch_size": 128,
            "policy_kwargs": dict(
                net_arch=dict(pi=[64, 64], qf=[64, 64])
            ),
        }


