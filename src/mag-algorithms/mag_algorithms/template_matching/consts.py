"""The Template matching constants."""
from enum import Enum

DEFAULT_CORR = -2
TOLERANCE = 1e-10


# Enums.
class Backend(Enum):
    """Enum Backend."""

    SKIMAGE = "skimage"
    OPENCV = "opencv"
