"""Method/Subclass of Interpolation: Cubic Spline."""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan

from .nearest import Nearest
from .base import Interpolation


class CubicSpline(Interpolation):
    """Scipy's CloughToucher2Dinterpolator."""

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        nearest_interpolated_scan = Nearest().interpolate(x, y, b, x_mesh, y_mesh)

        spline = CloughTocher2DInterpolator(list(zip(x, y)), b)
        spline_interpolated_b = spline(x_mesh, y_mesh)
        spline_mask = np.isnan(spline_interpolated_b)
        spline_interpolated_b[spline_mask] = nearest_interpolated_scan.b[spline_mask]

        return InterpolatedScan(x=x_mesh,
                                y=y_mesh,
                                b=spline_interpolated_b,
                                mask=nearest_interpolated_scan.mask,
                                interpolation_method=self.__class__.__name__)
