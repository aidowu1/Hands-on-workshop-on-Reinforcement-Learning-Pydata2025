from typing import Dict, Any
from collections import OrderedDict
import gymnasium as gym

from src.main.rl_algorithms.policy_gradient_sb.base_sb_algorithms import BasePolicyAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning import TD3HyperParameterTuning
from src.main.utility.utils import Helpers
import src.main.configs_rl as configs2

class TD3Algorithm(BasePolicyAlgorithm):
    """
    TD3 Algorithm
    """
    def __init__(
            self,
            env: gym.Env,
            env_name: str,
            hedging_type: HedgingType = HedgingType.gbm,
            total_timesteps: int = int(1e5),
            progress_bar: bool = True,
            is_normalize_obs: bool = False,
            reward_threshold = 1.0
    ):
        """
        Constructor
        :param env: Gym environment
        """
        super().__init__(
            env,
            env_name,
            hedging_type,
            total_timesteps,
            progress_bar,
            is_normalize_obs,
            reward_threshold)
        print(f"Start of RL {self.algorithm_type_name.upper()} agent learning for environment: {env_name}"
              f"and {hedging_type} hedging type")
        self.rl_problem_title = f"{self.algorithm_type_name.upper()} RL agent for {env_name} "

        self.createModelAndTensorboardPaths()
        self.createCallbacks()
        self.model = self.createModel()

    def preprocessHyperParameters(self) -> Dict[str, Any]:
        """
        Pre-processes the best hyperparameters for the RL algorithm
        :return: Pre-processed hyperparameters
        """
        # best_hyper_parameters_path = configs2.HYPER_PARAMETER_BEST_VALUES.format(self.algorithm_type_name)
        tuned_model_path = BaseHyperParameterTuning.createModelRootPath(self.algorithm_type, self.model_use_case)
        best_hyper_parameters_path = BaseHyperParameterTuning.createHyperparameterPath(tuned_model_path)
        best_hyper_parameters_all = Helpers.deserializeObject(best_hyper_parameters_path)
        self.noise_type = best_hyper_parameters_all[configs2.HYPER_PARAMETER_NOISE_TYPE]
        self.noise_std = best_hyper_parameters_all[configs2.HYPER_PARAMETER_NOISE_STD]

        best_net_architecture = TD3HyperParameterTuning.getNetArchHyperParameter()
        best_activation_function = TD3HyperParameterTuning.getActivationFunctionHyperParameter()
        net_arch = best_net_architecture[best_hyper_parameters_all[configs2.HYPER_PARAMETER_NET_ARCH]]
        activation_fn = best_activation_function[best_hyper_parameters_all[configs2.HYPER_PARAMETER_ACTIVATION_FN]]

        filter_list = [configs2.HYPER_PARAMETER_NOISE_TYPE, configs2.HYPER_PARAMETER_NOISE_STD,
                       configs2.HYPER_PARAMETER_NET_ARCH, configs2.HYPER_PARAMETER_ACTIVATION_FN]
        best_hyper_parameters = Helpers.filterDict(best_hyper_parameters_all, filter_list)
        policy_kwargs = {
            configs2.HYPER_PARAMETER_NET_ARCH: net_arch,
            configs2.HYPER_PARAMETER_ACTIVATION_FN: activation_fn,
        }
        best_hyper_parameters[configs2.HYPER_PARAMETER_POLICY_KWARGS] = policy_kwargs
        return best_hyper_parameters

    def untunedHyperParameters(self) -> Dict[str, Any]:
        """
        Untuned hyperparameters
        :return: Untuned hyperparameters
        """
        parameters = dict(
            OrderedDict
                (
                [
                    ('gamma', 0.98),
                    ('learning_rate', 0.001),
                ]
            )
        )
        return parameters

    @property
    def algorithm_type_name(self) -> str:
        """
        Getter of the type of RL algorithm
        :return: RL algorithm type
        """
        return RLAgorithmType.td3.name

    @property
    def algorithm_type(self) -> RLAgorithmType:
        """
        Getter of the type of RL algorithm
        :return: RL algorithm type
        """
        return RLAgorithmType.td3

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """
        Getter of the RL algorithm hyperparameters
        :return: Algorithm parameters
        """
        parameters = self.preprocessHyperParameters()
        return parameters



