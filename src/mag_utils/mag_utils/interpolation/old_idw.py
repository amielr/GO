"""Method/Subclass of Interpolation: old mag_utils idw."""
import numpy as np
from scipy.interpolate import Rbf
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from .base import Interpolation


class OldIdw(Interpolation):
    """Old mag_utils idw interpolation."""

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        idw = Rbf(x, y, b, function='inverse', mode='1-D')

        return InterpolatedScan(x=x_mesh,
                                y=y_mesh,
                                b=idw(x_mesh, y_mesh),
                                mask=self.calculate_mask(x, y, x_mesh, y_mesh),
                                interpolation_method=self.__class__.__name__)
