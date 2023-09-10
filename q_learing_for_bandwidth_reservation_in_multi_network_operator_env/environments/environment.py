from functools import lru_cache
import time
from typing import Any, Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
import gym
from gym import spaces
import pandas as pd

from utils.data import get_prices_predictions


class BookingEnv(gym.Env):
    """
    Custom environment for booking simulation using deep reinforcement learning.

    The RL-agent is provided with prices from various providers at each time step.
    It has to decide whether to accept the current offer or wait, and also predict
    which provider will offer the lowest price and when within the range.

    Attributes:
        prediction_model: Model used to predict prices.
        device (str): Torch device ('cuda' or 'cpu').
        current_timestamp: Current timestamp being considered.
        from_timestamp: Start timestamp for prediction.
        to_timestamp: End timestamp for prediction.
        step_seconds: Interval for each step in seconds.
        action_space (gym.spaces): Action space definition.
        observation_space (gym.spaces): Observation space definition.
        done (bool): Flag indicating if the episode is over.
        best_offer_price: Best offer price the agent encountered.
        large_penalty (float): Penalty to be applied.
        large_reward (float): Reward to be given for predicting the lowest price.
    """

    def __init__(
        self,
        from_timestamp: pd.Timestamp,
        to_timestamp: pd.Timestamp,
        step_seconds: int,
        decision_time_seconds: int,
        prediction_model: Any,
        device: str,
    ):
        super(BookingEnv, self).__init__()

        self.from_timestamp = from_timestamp
        self.to_timestamp = to_timestamp
        self.step_seconds = step_seconds
        self.decision_time_seconds = decision_time_seconds
        self.prediction_model = prediction_model
        self.device = device

        self.large_penalty = -50
        self.large_reward = 50
        all_prices = self._predict_prices()
        self.global_min_price = np.min(all_prices)
        self.global_min_price_index = np.argmin(all_prices)
        initial_prices = self._predict_prices()[0]
        self.num_providers = len(initial_prices)
        self.action_space = spaces.Discrete(self.num_providers + 1)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_providers,), dtype=np.float32
        )

        self.predictions_iterator = iter(self._predict_prices())

        self.current_timestamp = from_timestamp
        self.best_offer_info = {"price": float("inf"), "provider": None, "timestamp": None}
        self.done = False

    @lru_cache(maxsize=1)
    def _predict_prices(self) -> np.ndarray:
        return get_prices_predictions(
            self.prediction_model,
            (self.from_timestamp, self.to_timestamp, self.step_seconds),
            self.device
        ).cpu().numpy()

    def reset(self) -> np.ndarray:
        self.current_timestamp = self.from_timestamp
        self.done = False
        self.best_offer_info = {"price": float("inf"), "provider": None, "timestamp": None}
        return next(self.predictions_iterator)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        try:
            current_prices = next(self.predictions_iterator)
        except StopIteration:
            self.done = True
            current_prices = np.zeros(self.num_providers)

        if action < self.num_providers:
            chosen_price = current_prices[action]
            
            if action == self.global_min_price_index:
                reward = self.large_reward
            else:
                reward = self.global_min_price - chosen_price
            self.done = True
            
        elif action == self.num_providers:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.decision_time_seconds:
                reward = self.large_penalty
                self.done = True
            else:
                reward = -np.exp(0.01 * elapsed_time)

        return current_prices, reward, self.done, {}

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            predictions = self._predict_prices()
            from_unix = pd.Timestamp(self.from_timestamp).timestamp()
            to_unix = pd.Timestamp(self.to_timestamp).timestamp()
            timestamps_range = np.arange(
                from_unix, to_unix + self.step_seconds, self.step_seconds
            )
            datetime_range = [
                pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
                for ts in timestamps_range
            ]
            for i in range(predictions.shape[1]):
                plt.plot(datetime_range, predictions[:, i], label=f"Provider {i + 1}")
            plt.xticks(rotation=45)
            plt.xlabel("Time")
            plt.ylabel("Predicted Price")
            plt.title("Predicted Prices from Providers over Time")
            plt.legend()
            plt.tight_layout()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.show()

    def close(self) -> None:
        """
        Close the environment and perform any necessary cleanup.
        """
        plt.close()
