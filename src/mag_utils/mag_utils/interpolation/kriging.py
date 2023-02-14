"""Method/Subclass of Interpolation: Kriging(GaussianProcessRegressor)."""
import numpy as np
from pykrige.ok import OrdinaryKriging
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from .base import Interpolation


class Kriging(Interpolation):
    """Kriging (PyKrige) interpolation."""

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        kriging = OrdinaryKriging(x,
                                  y,
                                  b,
                                  variogram_model="linear",
                                  verbose=False,
                                  enable_plotting=False)

        interpolated_b, _ = kriging.execute("grid", x_mesh[0], y_mesh[:, 0])

        return InterpolatedScan(x=x_mesh,
                                y=y_mesh,
                                b=interpolated_b.reshape(x_mesh.shape),
                                mask=self.calculate_mask(x, y, x_mesh, y_mesh),
                                interpolation_method=self.__class__.__name__)
