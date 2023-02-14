import numpy as np

from mag_utils.interpolation.registry import interpolation_registry
from mag_utils.scans.interpolated_scan import InterpolatedScan


def test_interpolation_methods():
    for interp_name, interp_method in interpolation_registry.items():
        # Set up interpolation method.
        interp = interp_method()

        # Set the points (with values)
        x = np.array([10, 20., 30])
        y = np.array([20, 30., 10])
        b = np.array([40, 10., 0])

        # Calculate the x_mesh,y_mesh area. (grid)
        x_mesh, y_mesh = interp.calculate_xy_mesh(x=x, y=y, distance_between_points=0.5)

        # Interpolate the area by the points and their b.
        interpolated_scan = interp.interpolate(x, y, b, x_mesh, y_mesh)

        assert isinstance(interpolated_scan,
                          InterpolatedScan), \
            f"{interp_name} didn't return InterpolatedScan. got {type(interpolated_scan)}"
        assert interpolated_scan.b.shape == x_mesh.shape, \
            f"{interp_name}'s interpolated b has wrong shape. got {interpolated_scan.b.shape}"
        assert interpolated_scan.mask.shape == x_mesh.shape, \
            f"{interp_name}'s interpolated mask has wrong shape. got {interpolated_scan.b.shape}"
        assert interpolated_scan.interpolation_method == interp_name, \
            f"{interp_name}'s interpolation_method is not correct. got {interpolated_scan.interpolation_method} " \
            f"instead of {interp_name}"
