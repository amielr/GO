"""Method/Subclass of Interpolation: Nearest."""
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from .base import Interpolation


class Nearest(Interpolation):
    """Scipy's Nearest neighbor interpolation."""

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        nearest = NearestNDInterpolator(list(zip(x, y)), b)

        return InterpolatedScan(x=x_mesh,
                                y=y_mesh,
                                b=nearest(x_mesh, y_mesh),
                                mask=self.calculate_mask(x, y, x_mesh, y_mesh),
                                interpolation_method=self.__class__.__name__)
