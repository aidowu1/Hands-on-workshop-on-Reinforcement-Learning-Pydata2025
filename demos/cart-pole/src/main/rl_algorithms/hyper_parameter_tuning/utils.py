from typing import Any
from torch import nn as nn


class Helpers:
    """
    Helper functions for hyperparameter tuning.
    """
    @staticmethod
    def convertOffPolicyParams(sampled_params: dict[str, Any]) -> dict[str, Any]:
        hyperparams = sampled_params.copy()

        hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
        del hyperparams["one_minus_gamma"]

        net_arch = sampled_params["net_arch"]
        del hyperparams["net_arch"]

        for name in ["batch_size"]:
            if f"{name}_pow" in sampled_params:
                hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
                del hyperparams[f"{name}_pow"]

        net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
            "large": [256, 256, 256],
            "verybig": [512, 512, 512],
        }[net_arch]

        if "train_freq" in sampled_params:
            # Update to data ratio of 1, for n_envs=1
            hyperparams["gradient_steps"] = sampled_params["train_freq"]

            if "subsample_steps" in sampled_params:
                hyperparams["gradient_steps"] = max(sampled_params["train_freq"] // sampled_params["subsample_steps"],
                                                    1)
                del hyperparams["subsample_steps"]

        hyperparams["policy_kwargs"] = hyperparams.get("policy_kwargs", {})
        hyperparams["policy_kwargs"]["net_arch"] = net_arch

        if "activation_fn" in sampled_params:
            activation_fn_name = sampled_params["activation_fn"]
            del hyperparams["activation_fn"]

            activation_fn = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "elu": nn.ELU,
                "leaky_relu": nn.LeakyReLU,
            }[activation_fn_name]
            hyperparams["policy_kwargs"]["activation_fn"] = activation_fn

        # TQC/QRDQN
        if "n_quantiles" in sampled_params:
            del hyperparams["n_quantiles"]
            hyperparams["policy_kwargs"].update({"n_quantiles": sampled_params["n_quantiles"]})

        return hyperparams