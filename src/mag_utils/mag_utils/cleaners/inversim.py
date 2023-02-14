from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Sequence

from scipy.sparse.linalg import svds
from tqdm.contrib import tzip

from sklearn.linear_model import Ridge
import numpy as np

from mag_utils.algo_tools.convex_hull_grids import convex_hull_grid, convex_hull_grid_at_depths, \
    rectangular_convex_hull_grid
from mag_utils.algo_tools.simulations import simulate_b_ax_from_dipole, find_moments_with_orthogonal_simulations
from mag_utils.scans import HorizontalScan


@dataclass(init=True, repr=True)
class DepthInterval:
    min_depth: float
    max_depth: float
    dipoles_spacing: float
    zero_out: bool = False  # if dipoles should be simulated in this interval, but be zeroed out in the final result


# pylint: disable-msg=too-many-locals
def get_simulation_matrices(scan: HorizontalScan,
                            depth_intervals: Union[DepthInterval, Sequence[DepthInterval]],
                            is_rectangular=False,
                            sampling_positions=None,
                            height_above_sensor=0.,
                            samples_spacing=1.,
                            dipoles_convex_hull_factor=1.3,
                            b_e_direction=np.array([0., 1., -1.])):
    """
    This is a helper method for the inversim. It computes the simulation matrix used for the ridge regression and the
    matrix later used to simulate the interpolation grid from the coefs of the regression.
    Args:
        scan: HorizontalScan object.
        depth_intervals: depth intervals to simulate dipoles at.
        is_rectangular:  If True, the sampled grid will be rectangular.
        sampling_positions: None or np.ndarray of shape (#samples, 3). Grid of interpolation/extrapolation.
        height_above_sensor: If sampling position is None, this will be the height of the interpolated grid above the
        scan's mean height.
        samples_spacing: spacing of the interpolated / extrapolated data.
        dipoles_convex_hull_factor: how far from scan's x,y you want to simulate dipoles
        b_e_direction: assumed earth's magnetic field's direction, shape (1,3).

    Returns: simulation mats of the dipoles for the scan and sampled grid and the x, y, z of the sampled grid.
    given the flatten of the moments of the dipoles moments m simulation_mat@m is the simulation of the magnetic field
    at the scan / interpolated grid of the dipoles.
    """

    z_mean = scan.a.mean()

    # Scan position matrix.
    scan_positions = np.column_stack([scan.x, scan.y, scan.a])

    rectangle_shape = None  # shape of the returned rectangular grid if is_rectangular.

    # Interpolation positions matrix.
    if sampling_positions is None:
        if is_rectangular:
            x_interp, y_interp, n_major, n_minor = rectangular_convex_hull_grid(scan.x, scan.y, samples_spacing)
            rectangle_shape = (n_major, n_minor)
        else:
            x_interp, y_interp = convex_hull_grid(scan.x, scan.y, samples_spacing)
        z_interp = np.ones_like(x_interp) * (z_mean + height_above_sensor)
        sampling_positions = np.column_stack([x_interp, y_interp, z_interp])

    # Check if interpolation grid is the same as the scan itself to reduce computations.
    is_interp_mat_equals_origin_mat = np.array_equal(scan_positions.ravel(), sampling_positions.ravel())

    if isinstance(depth_intervals, DepthInterval):
        depth_intervals = [depth_intervals]

    # Compute the dipole positions for each depth interval.
    # ----------------------------------------------------------------------------------------------------
    zero_out_dipoles = []  # A boolean list that says if the dipole should be zero out after optimization.
    dipoles_positions = []  # Positions of all the dipoles to simulate.
    for depth_interval in depth_intervals:
        # Compute the position of the dipoles at the current depth interval:
        # --------------------------------------------------------------------------------------
        depth1 = z_mean - depth_interval.max_depth
        depth2 = z_mean - depth_interval.min_depth
        dipole_spacing = depth_interval.dipoles_spacing
        depths = np.linspace(min(depth1, depth2), max(depth1, depth2),
                             int(abs(depth1 - depth2) / dipole_spacing) + 1)
        dipoles_x, dipoles_y, dipoles_z = convex_hull_grid_at_depths(scan.x,
                                                                     scan.y,
                                                                     depths,
                                                                     dipole_spacing,
                                                                     dipoles_convex_hull_factor)
        # --------------------------------------------------------------------------------------

        # Update the whole dipoles position list and zero out dipoles list.
        # -------------------------------------------------------------------------
        current_positions = np.column_stack([dipoles_x, dipoles_y, dipoles_z])
        zero_out_dipoles.extend([depth_interval.zero_out] * len(current_positions))
        dipoles_positions.extend(list(current_positions))
        # -------------------------------------------------------------------------

    dipoles_positions = np.array(dipoles_positions)
    # ----------------------------------------------------------------------------------------------------

    # Compute 3 orthonormal simulations per dipole that spans all the simulations that can be created
    # by a dipole at that location, and compute those simulations in the interpolation grid if they are not zeroed out.
    # -----------------------------------------------------------------------------------------------------------------
    scan_sim_mat = []
    interp_sim_mat = []
    for point, zero_out in tzip(dipoles_positions, zero_out_dipoles):
        moment1, moment2, moment3, sim1, sim2, sim3 = \
            find_moments_with_orthogonal_simulations(point,
                                                     b_e_direction,
                                                     scan_positions,
                                                     remove_dc=True)
        scan_sim_mat.extend([sim1, sim2, sim3])

        if not is_interp_mat_equals_origin_mat:
            if not zero_out:
                sim1 = simulate_b_ax_from_dipole(point, moment1, b_e_direction, sampling_positions)
                sim2 = simulate_b_ax_from_dipole(point, moment2, b_e_direction, sampling_positions)
                sim3 = simulate_b_ax_from_dipole(point, moment3, b_e_direction, sampling_positions)
                interp_sim_mat.extend([sim1, sim2, sim3])
            else:
                interp_sim_mat.extend([np.zeros(len(sampling_positions))] * 3)

    scan_sim_mat = np.array(scan_sim_mat).T
    # -----------------------------------------------------------------------------------------------------------------

    # Give each parameter a weight such that each x,y,z location has the same l2 penalty for
    # enlarging the scan's variance.
    # --------------------------------------------------------------------------------------
    weights = 1 / np.sum(np.abs(scan_sim_mat.T @ scan_sim_mat), axis=1)
    scan_sim_mat = scan_sim_mat * weights
    # --------------------------------------------------------------------------------------

    # Normalize the scan sim matrix by it's largest singular value.
    # -------------------------------------------------------------
    max_s = svds(scan_sim_mat, k=1)[1][0]
    scan_sim_mat /= max_s
    # -------------------------------------------------------------

    # Compute and the interpolation simulation matrix.
    # --------------------------------------------------------
    if is_interp_mat_equals_origin_mat:
        zero_out = np.array(~np.repeat(zero_out_dipoles, 3), dtype=int)
        interp_sim_mat = scan_sim_mat * zero_out
    else:
        interp_sim_mat = np.array(interp_sim_mat).T * weights
        interp_sim_mat = interp_sim_mat / max_s
    # --------------------------------------------------------

    return (scan_sim_mat,
            interp_sim_mat,
            sampling_positions[:, 0],
            sampling_positions[:, 1],
            sampling_positions[:, 2],
            dipoles_positions,
            rectangle_shape)


# pylint: disable-msg=too-many-locals
def inversim(scan: HorizontalScan,
             depth_intervals: Union[DepthInterval, Sequence[DepthInterval]],
             dipoles_convex_hull_factor: float = 1.3,
             samples_spacing: float = 1.,
             l2_penalty: float = .1,
             lock_scan_position=False,
             is_rectangular=False,
             b_e_direction=None,
             sampling_positions=None,
             height_above_sensor=0.):
    """
    Args:
        scan: HorizontalScan object.
        depth_intervals: depth intervals to simulate dipoles at.
        dipoles_convex_hull_factor: how far from scan's x,y you want to simulate dipoles.
        samples_spacing: spacing of the interpolated grid. Ignored if lock_scan_position == True.
        l2_penalty: ridge penalty of the regression that computes the coefs each simulation is multiplied by.
        lock_scan_position: If set to true, no interpolation happens. Only cleanning of the original scan.
        is_rectangular: If True, the sampled grid will be rectangular.
        b_e_direction: Direction of Earth's magnetic field. If None [0, 1, -1] is assumed. Shape (1,3).
        sampling_positions: The position of the interpolated/extrapolated points. If None, a grid with constant height
        and a spacing of samples_spacing will be created. Shape = (num_samples,3).
        height_above_sensor: Height of the interpolated grid above the mean height of the sensor.

    Returns: New cleaned HorizontalScan.
    """

    if b_e_direction is None:
        b_e_direction = np.array([0., 1., -1.])
    if lock_scan_position:
        sampling_positions = np.column_stack([scan.x, scan.y, scan.a])

    # Generate simulation matrices for the scan positions and interpolated positions.
    (scan_sim_mat, interp_sim_mat,
     x_interp, y_interp, z_interp,
     _, rectangle_shape) = get_simulation_matrices(scan,
                                                   depth_intervals,
                                                   is_rectangular,
                                                   sampling_positions,
                                                   height_above_sensor,
                                                   samples_spacing,
                                                   dipoles_convex_hull_factor,
                                                   b_e_direction)

    # Use Ridge Regression to estimate the coefficients of each simulation.
    regr = Ridge(l2_penalty)
    b = scan.b - scan.b.mean()
    regr.fit(scan_sim_mat, b)

    # Compute by how much the ridge regression dropped the coefficients.
    cleaned_orig_b = scan_sim_mat @ regr.coef_
    ridge_attenuation = cleaned_orig_b / np.linalg.norm(cleaned_orig_b) ** 2 @ b

    # Simulate the clean scan based on the regression coefs.
    b_interp = (interp_sim_mat @ regr.coef_) * ridge_attenuation

    new_scan = deepcopy(scan)
    new_scan.x, new_scan.y, new_scan.a, new_scan.b = x_interp, y_interp, z_interp, b_interp

    if is_rectangular:
        return new_scan, rectangle_shape

    return new_scan
