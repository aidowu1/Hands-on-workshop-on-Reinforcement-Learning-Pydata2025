import gymnasium as gym
from typing import Dict, Any
import torch.nn as nn

from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.rl_algorithms.train_evaluate_test.save_on_base_reward_callback import SaveOnBestTrainingRewardCallback
from src.main.rl_algorithms.train_evaluate_test.base_algorithms import BaseRLAlgorithm
from src.main.rl_algorithms.hyper_parameter_tuning.ppo_hyper_parameter_tuning import PPOHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers

class PPOTrainAlgorithm(BaseRLAlgorithm):
    """
    TD3 Algorithm
    """

    def __init__(
            self,
            env: gym.Env,
            rl_problem_title: str = "RL Delta Hedger",
            hedging_type: HedgingType = HedgingType.gbm,
            max_steps: int = configs2.N_STEPS * configs2.N_EPISODES,
            check_freq: int = configs2.CHECKPOINT_FREQ,
            model_use_case: str = None
    ):
        """
        
        :param env: RL environment
        :param rl_algorithm_type:
        :param rl_problem_title: 
        :param max_steps: 
        """
        super(PPOTrainAlgorithm, self).__init__(
            env,
            RLAgorithmType.ppo,
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
        self._model.learn(total_timesteps=self._max_train_steps, callback=callback)

    @property
    def tuned_hyper_parameters(self) -> Dict[str, Any]:
        """
        Gets and pre-processes the best tuned hyperparameters for the RL algorithm
        :return: Best hyperparameters
        """
        #best_hyper_parameters_path = configs2.HYPER_PARAMETER_BEST_VALUES.format(self._rl_algo_type.name)
        best_hyper_parameters_path = BaseHyperParameterTuning.createHyperparameterPath(self._tuned_model_root_path)
        best_hyper_parameters_all = Helpers.deserializeObject(best_hyper_parameters_path)

        best_net_architecture = PPOHyperParameterTuning.getNetArchHyperParameter()
        best_activation_function = PPOHyperParameterTuning.getActivationFunctionHyperParameter()
        net_arch = best_net_architecture[best_hyper_parameters_all[configs2.HYPER_PARAMETER_NET_ARCH]]
        activation_fn = best_activation_function[best_hyper_parameters_all[configs2.HYPER_PARAMETER_ACTIVATION_FN]]
        ortho_init = best_hyper_parameters_all[configs2.HYPER_PARAMETER_ORTHO_INIT]
        log_std_init = best_hyper_parameters_all[configs2.HYPER_PARAMETER_LOG_STD_INIT]

        filter_list = [configs2.HYPER_PARAMETER_LR_SCHEDULE,
                       configs2.HYPER_PARAMETER_ORTHO_INIT,
                       configs2.HYPER_PARAMETER_LOG_STD_INIT,
                       configs2.HYPER_PARAMETER_NET_ARCH,
                       configs2.HYPER_PARAMETER_ACTIVATION_FN]

        best_hyper_parameters = Helpers.filterDict(best_hyper_parameters_all, filter_list)
        policy_kwargs = {
            configs2.HYPER_PARAMETER_NET_ARCH: net_arch,
            configs2.HYPER_PARAMETER_ACTIVATION_FN: activation_fn,
            configs2.HYPER_PARAMETER_ORTHO_INIT: ortho_init,
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
        return {
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "ent_coef": 0.0,
            "sde_sample_freq": 4,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "learning_rate": 3e-5,
            "use_sde": True,
            "policy_kwargs": {
                "log_std_init": -2.7,
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [256, 256],
                             "vf": [256, 256]}
        }
}