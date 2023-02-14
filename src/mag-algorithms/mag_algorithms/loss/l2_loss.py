"""L2 Loss."""
import numpy as np
from .loss import Loss


class L2Loss(Loss):
    """
    L2 loss.

    It is defined as: L(a) = a^2.
    """

    def _compute_loss(self, input: np.array, target: np.array) -> np.array:
        return np.linalg.norm(input - target, ord=2, axis=-1)
