"""Method/Subclass of Interpolation: old mag_utils.mag_utils.mag_utils.mag_utils v21 interpolation."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic as radQ
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from .base import Interpolation


class OldV21(Interpolation):
    """Old mag_utils.mag_utils.mag_utils.mag_utils v21 interpolation."""

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        # Set the regressor.
        kernel = radQ(length_scale=1.2, alpha=0.78) * RBF(10, (1e-2, 1e2))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.1, normalize_y=False)

        # Fit the data.
        gaussian_process.fit(np.stack([x, y], axis=1), b)

        # Return the predicted values in the area.
        return InterpolatedScan(x=x_mesh,
                                y=y_mesh,
                                b=gaussian_process.predict(
                                    np.stack([x_mesh.flatten(), y_mesh.flatten()], axis=1)).reshape(x_mesh.shape),
                                mask=self.calculate_mask(x, y, x_mesh, y_mesh),
                                interpolation_method=self.__class__.__name__)
