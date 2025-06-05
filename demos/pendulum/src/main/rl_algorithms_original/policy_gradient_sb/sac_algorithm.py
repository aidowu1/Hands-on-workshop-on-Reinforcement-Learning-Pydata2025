from typing import Dict, Any
import gymnasium as gym

from src.main.rl_algorithms.policy_gradient_sb.base_sb_algorithms import BasePolicyAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.rl_algorithms.hyper_parameter_tuning.sac_hyper_parameter_tuning import SACHyperParameterTuning
from src.main.utility.utils import Helpers
import src.main.configs_rl as configs2

class SACAlgorithm(BasePolicyAlgorithm):
    """
    SAC Algorithm
    """
    def __init__(
            self,
            env: gym.Env,
            env_name: str,
            hedging_type: HedgingType = HedgingType.gbm,
            total_timesteps: int = int(1e5),
            progress_bar: bool = True,
            is_normalize_obs: bool = False,
            reward_threshold: float = -200.0,
            model_use_case: str = None
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
            reward_threshold,
            model_use_case
        )
        print(f"Start of RL {self.algorithm_type_name} agent learning for environment: {env_name}"
              f" and {self.hedging_type} hedging type")
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

        best_net_architecture = SACHyperParameterTuning.getNetArchHyperParameter()
        best_activation_function = SACHyperParameterTuning.getActivationFunctionHyperParameter()
        net_arch = best_net_architecture[best_hyper_parameters_all[configs2.HYPER_PARAMETER_NET_ARCH]]
        activation_fn = best_activation_function[best_hyper_parameters_all[configs2.HYPER_PARAMETER_ACTIVATION_FN]]
        log_std_init = best_hyper_parameters_all[configs2.HYPER_PARAMETER_LOG_STD_INIT]

        filter_list = [
                       configs2.HYPER_PARAMETER_LOG_STD_INIT,
                       configs2.HYPER_PARAMETER_NET_ARCH,
                       configs2.HYPER_PARAMETER_ACTIVATION_FN]

        best_hyper_parameters = Helpers.filterDict(best_hyper_parameters_all, filter_list)
        policy_kwargs = {
            configs2.HYPER_PARAMETER_NET_ARCH: net_arch,
            configs2.HYPER_PARAMETER_ACTIVATION_FN: activation_fn,
            configs2.HYPER_PARAMETER_LOG_STD_INIT: log_std_init,
        }
        best_hyper_parameters[configs2.HYPER_PARAMETER_POLICY_KWARGS] = policy_kwargs
        return best_hyper_parameters

    @property
    def algorithm_type_name(self) -> str:
        """
        Getter of the type of RL algorithm
        :return: RL algorithm type
        """
        return RLAgorithmType.sac.name

    @property
    def algorithm_type(self) -> RLAgorithmType:
        """
        Getter of the type of RL algorithm
        :return: RL algorithm type
        """
        return RLAgorithmType.sac

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """
        Getter of the RL algorithm hyperparameters
        :return: Algorithm parameters
        """
        parameters = self.preprocessHyperParameters()
        return parameters

