import numpy as np


def calibration(bx, by, bz):
    """
    Get bx, by, bz and calibrate the value.

    Args:
        bx: magnetic field in axis x.
        by: magnetic field in axis y.
        bz: magnetic field in axis z.

    Returns:
        Calibrated b.
    """
    return np.linalg.norm(np.stack([bx, by, bz]), axis=0)
