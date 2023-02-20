"""Base class for interpolation methods."""
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan


class Interpolation:
    """
    Base interpolation class for all interpolation methods.

    Examples:
        # Import the registry
            from mag_utils.mag_utils.mag_utils.mag_utils.interpolation.registry import interpolation_registry

        # Set up interpolation method.
            interp = interpolation_registry["RBF"]()

        # Calculate the x_mesh,y_mesh area.
            x_mesh, y_mesh = interp.calculate_xy_mesh(x=x, y=y, distance_between_points=0.1)

        # Interpolate the area by the points and their b, and the mask.
            nb = interp.interpolate(x, y, b, x_mesh, y_mesh).reshape(x_mesh.shape)

            mask = interp.calculate_mask(x, y, x_mesh, y_mesh).reshape(x_mesh.shape)

        # Cut the new interpolated area by the mask.
            nb[~mask] = np.nan

        # Plot/Draw the interpolated (after cut) shape.
            fig, ax = plt.subplots(1)

            interp.draw_image(ax, x, y, nb, b)
    """

    def interpolate(self, x: np.ndarray, y: np.ndarray, b: np.ndarray, x_mesh: np.ndarray,
                    y_mesh: np.ndarray) -> InterpolatedScan:
        """
        Activate the interpolation on x, y, b in x_mesh, y_mesh.

        Args:
            x: X coordinate of the points. [N]
            y: Y coordinate of the points. [N]
            b: The Field-Size (b) of the x,y points. [N]
            x_mesh: The x_mesh-arange of the area. [M, K]
            y_mesh: The y_mesh-arange of the area. [M, K]

        Returns:
            Interpolated scan, after intepolation without cutting - shape (x_mesh, y_mesh) interpolated block.
        """
        raise NotImplementedError()

    def calculate_xy_mesh(self, x: np.ndarray, y: np.ndarray, distance_between_points: float):
        """
        Calculate the aranged area of the points (the smallest area that includes all the points).

        Args:
            distance_between_points: distance between points in meters.
            x: X coordinate of the points.
            y: Y coordinate of the points.

        Returns:
            x_mesh, y_mesh.
        """
        x_arrange = np.arange(x.min(), x.max(), distance_between_points)
        y_arrange = np.arange(y.min(), y.max(), distance_between_points)
        x_mesh, y_mesh_vertical = np.meshgrid(x_arrange, y_arrange)
        y_mesh = np.flipud(y_mesh_vertical)

        return x_mesh, y_mesh

    def calculate_mask(self, x: np.ndarray, y: np.ndarray, x_mesh: np.ndarray, y_mesh: np.ndarray):
        """
        Calculate which mesh points are interpolated and which are extrapolate.

        Args:
            x: The X points [N].
            y: The Y points [N].
            x_mesh: The x_mesh area-points [M, K].
            y_mesh: The y_mesh area-points [M, K].

        Returns:
             A boolean matrix of shape [same shape as x_mesh] which indicates
             rather each point is interpolated or extrapolated.
        """
        # By convex-hull, find the edges-points.
        convex_hull = ConvexHull(np.stack([x, y], axis=1))
        pairs = np.array([y[convex_hull.vertices], x[convex_hull.vertices]])

        # By points_in_poly we find which points are inside the
        # polygon created by the edges from the convex-hull.
        polygon = Path(pairs.T)
        mask = polygon.contains_points(np.stack([y_mesh.flatten(), x_mesh.flatten()], axis=1))

        return mask.reshape(x_mesh.shape)
