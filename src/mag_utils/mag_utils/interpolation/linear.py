"""Method/Subclass of Interpolation: Linear."""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from .nearest import Nearest
from .base import Interpolation


class Linear(Interpolation):
    """Scipy's LinearNDInterpolator."""

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        nearest_interpolated_scan = Nearest().interpolate(x, y, b, x_mesh, y_mesh)

        linear = LinearNDInterpolator(list(zip(x, y)), b)
        linear_interpolated_b = linear(x_mesh, y_mesh)
        linear_mask = np.isnan(linear_interpolated_b)
        linear_interpolated_b[linear_mask] = nearest_interpolated_scan.b[linear_mask]

        return InterpolatedScan(x=x_mesh,
                                y=y_mesh,
                                b=linear_interpolated_b,
                                mask=nearest_interpolated_scan.mask,
                                interpolation_method=self.__class__.__name__)
