"""Huber loss."""
import numpy as np
from .loss import Loss


class HuberLoss(Loss):
    """
    Huber loss.

    It is defined as: L(a) = a^2 / 2 if |a| <= delta and delta * (|a| - delta/2) otherwise.
    """

    def __init__(self, reduction: str = 'mean', delta: float = 1.0):
        """
        Initialize a huber loss.

        Args:
            reduction: reduction method as in Loss
            delta: delta parameter for the huber loss.
        """
        super().__init__(reduction)
        self.delta = delta

    def _compute_loss(self, input: np.array, target: np.array) -> np.array:
        huber_mse = 0.5 * (input - target) ** 2
        huber_mae = self.delta * (np.abs(input - target) - 0.5 * self.delta)
        return np.sum(np.where(np.abs(input - target) <= self.delta, huber_mse, huber_mae), axis=-1)
