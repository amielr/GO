"""The mag_utils.mag_utils.mag_utils.mag_utils constants."""
from typing import Union, List
import numpy as np

# The version of mag_utils.mag_utils.mag_utils.mag_utils package
mag_UTILS_VERSION = "3.2"

# Set an object that include the types of the supported sequences' types.
Sequence = Union[List, np.ndarray]


def docs_variables_decorator(*sub):
    """
    Decorate function that add variables into docstrings.

    Args:
        *sub: Any variables than you want to use in your docstring.

    Returns:
        The docstring after inserting the variables.
    """

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec
