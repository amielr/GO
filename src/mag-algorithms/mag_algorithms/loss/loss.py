"""General loss abstract base class."""
import numpy as np
from abc import ABC
from typing import Union


class Loss(ABC):
    """
    Abstract base class of losses.

    This was defined to be parralel to the torch definiton of loss functions.
    """

    reduction: str

    def __init__(self, reduction: Union[str, None] = 'mean') -> None:
        """
        Initialize a loss function.

        Args:
            reduction: way to reduce the loss over the batch. options are ['mean', 'sum', 'none']
        """
        super(ABC, self).__init__()
        self.reduction = reduction
        allowed_reductions = ['mean', 'sum', 'none', None]
        if reduction not in allowed_reductions:
            raise ValueError(f'Reduction not in allowed reductions: {allowed_reductions}')

    def reduce_according_to_str(self, losses: np.array):
        """
        Reduce the batch results according to the reduction variable.

        Args:
            losses: loss batch

        Returns:
            np.array of reduced losses.
        """
        if self.reduction == 'mean':
            return np.mean(losses, axis=-1)
        elif self.reduction == 'sum':
            return np.sum(losses, axis=-1)
        elif self.reduction == 'none' or self.reduction is None:
            return losses

    def forward(self, input: np.array, target: np.array) -> np.array:
        """
        Compute the loss over the input and target arrays.

        Args:
            input: np.array of inputs.
            target: np.array of targets.

        Returns:
            np.array of resulting losses, reduced according to the reduction variable.
        """
        return self.reduce_according_to_str(self._compute_loss(input, target))

    def _compute_loss(self, input: np.array, target: np.array) -> np.array:
        """
        Compute loss without reducing.

        Args:
            input: np.array of inputs.
            target: np.array of targets.

        Returns:
            np.array of resulting losses, before reduction.
        """
        raise NotImplementedError()
