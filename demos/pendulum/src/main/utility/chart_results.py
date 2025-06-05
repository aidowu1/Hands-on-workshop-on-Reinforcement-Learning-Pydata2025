import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Dict, Tuple, Any, Optional

mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["font.size"] = 16

class ChartResults:
    """
    Component used to chat the results of an RL agent
    """

    @staticmethod
    def plotRewardCurve(
            global_rewards: List[float],
            window_size: int
    ):
        """
        Plot reward curves
        :param global_rewards: Global rewards
        :param window_size: Window size
        :return: None
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
        rewards_x, rewards_y = ChartResults.computeMovingAverage(global_rewards, window_size)
        axes[0].plot(global_rewards)
        axes[0].set_title("RL Agent's Reward Curve")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative Reward")

        axes[1].plot(rewards_x, rewards_y)
        axes[1].set_title(f"Moving average over {window_size} episodes")
        axes[1].set_xlabel("Total Number of Training Episodes")
        axes[1].set_ylabel("Avg. Reward")
        axes[1].grid()
        plt.show()


    @staticmethod
    def computeMovingAverage(data: List[float], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute moving average
        :param data: List of floats
        :param window_size: Window size
        :return: Moving average transformed data
        """
        window = np.ones(int(window_size)) / float(window_size)
        x = np.arange(0, len(data))
        y = np.convolve(data, window, "same")
        x_ = x[:-window_size]
        y_ = y[:-window_size]
        return x_, y_



