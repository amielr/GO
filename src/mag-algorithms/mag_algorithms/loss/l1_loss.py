"""L1 Loss."""
import numpy as np
from .loss import Loss


class L1Loss(Loss):
    """
    L1 loss.

    It is defined as: L(a) = |a|.
    """

    def _compute_loss(self, input: np.array, target: np.array) -> np.array:
        return np.linalg.norm(input - target, ord=1, axis=-1)
