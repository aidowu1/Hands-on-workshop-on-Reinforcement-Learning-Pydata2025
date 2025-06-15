import gymnasium as gym
from typing import Dict, Any

from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.rl_algorithms.train_evaluate_test.save_on_base_reward_callback import SaveOnBestTrainingRewardCallback
from src.main.rl_algorithms.train_evaluate_test.base_algorithms import BaseRLAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.sac_hyper_parameter_tuning import SACHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers

class SACTrainAlgorithm(BaseRLAlgorithm):
    """
    TD3 Algorithm
    """

    def __init__(
            self,
            env: gym.Env,
            rl_problem_title: str = "RL Delta Hedger",
            hedging_type: HedgingType = HedgingType.gbm,
            max_steps: int = int(1e4),
            check_freq: int = configs2.CHECKPOINT_FREQ,
            model_use_case: str = None
    ):
        """
        Constructor
        :param env: RL environment
        :param rl_algorithm_type:
        :param rl_problem_title: 
        :param max_steps:
        :param check_freq:
        :param model_use_case:
        """
        super(SACTrainAlgorithm, self).__init__(
            env,
            RLAgorithmType.sac,
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
        callback = SaveOnBestTrainingRewardCallback(check_freq=self._check_freq, log_dir=self._log_dir)
        self._model.learn(total_timesteps=self._max_train_time_steps, callback=callback)

    @property
    def tuned_hyper_parameters(self) -> Dict[str, Any]:
        """
        Gets and pre-processes the best tuned hyperparameters for the RL algorithm
        :return: Best hyperparameters
        """
        # best_hyper_parameters_path = configs2.HYPER_PARAMETER_BEST_VALUES.format(self._rl_algo_type.name)
        best_hyper_parameters_path = BaseHyperParameterTuning.createHyperparameterPath(self._tuned_model_root_path)
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
    def non_tuned_hyperparameters(self) -> Dict[str, Any]:
        """
        Getter for the non-turned hyperparameters
        :return:
        """
        params = {
            'gamma': 0.9,
            'learning_rate': 0.004843759072455456,
            'batch_size': 1024,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'train_freq': 16,
            'tau': 0.005,
            'policy_kwargs': dict(log_std_init=0.2571311109643921, net_arch=[64, 64])
        }

        # return {
        #     "gamma": 1.0,
        #     "learning_rate": 0.0001,
        #     "batch_size": 1024,
        #     "buffer_size": 16384,
        #     #"learning_starts": 5000,
        #     "policy_kwargs": dict(net_arch=[64, 64, 64])
        #     }
        return params


