import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from mag_utils.cleaners.directional_noise_removal import remove_rotating_dipoles_from_rotations, \
    minimum_rotation_between_3d_vectors
from mag_utils.cleaners.inversim import inversim, DepthInterval
from mag_utils.scans import HorizontalScan


def auto_magen_cleaning(scan: HorizontalScan,
                        raw_scan=None,
                        dipole_spacing=4,
                        depth_search_interval=(5., 30.),
                        l2_penalty=1.,
                        height_above_sensor=0.,
                        is_rectangular=False,
                        show_intermediate_plots=False):
    """
    Clean magen file by first removing the affect of the rotating dipoles by gps rotation estimation, then use
    ridge regression on simulations below ground to clean the scan and interpolate it.

    Args:
        scan: HorizontalScan object.
        raw_scan: The original raw scan if one exists, it uses the original scan to compute the direction of the walk in
        each sample even if certain samples were removed from the new scan.
        dipole_spacing: Spacing of the grid of the dipoles that are simulated in the inversim / ridge regression.
        depth_search_interval: From which depth and to which depth below sensor to simulate the dipoles.
        l2_penalty: l2 penalty of the ridge regression in the inversim.
        height_above_sensor: At which height above the sensor to simulate the clean scan after the ridge regression
        (i.e., z-extrapolation).
        is_rectangular: If true the shape of the returned scan is rectangular.
        show_intermediate_plots: If true, the original scan, the scan after directional noise removal and the scan after
        inversim will be plotted.

    Returns: new clean interpolated scan.
    """
    if show_intermediate_plots:
        plt.tricontourf(scan.x, scan.y, scan.b, levels=20)
        plt.colorbar()
        plt.title("Original Scan")
        plt.scatter(scan.x, scan.y, s=0.1, c="black")
        plt.axis("equal")
        plt.show()

    scan = clean_rotating_dipoles_by_gps(scan, raw_scan)

    if show_intermediate_plots:
        plt.tricontourf(scan.x, scan.y, scan.b, levels=20)
        plt.colorbar()
        plt.title("Scan after GPS Cleaning")
        plt.scatter(scan.x, scan.y, s=0.1, c="black")
        plt.axis("equal")
        plt.show()

    new_scan = inversim(scan,
                        DepthInterval(depth_search_interval[0],
                                      depth_search_interval[1],
                                      dipoles_spacing=dipole_spacing),
                        l2_penalty=l2_penalty,
                        height_above_sensor=height_above_sensor,
                        is_rectangular=is_rectangular)

    if show_intermediate_plots:
        plt.tricontourf(new_scan.x,
                        new_scan.y,
                        new_scan.b,
                        levels=20)
        plt.colorbar()
        plt.title("Scan after Inversim")
        plt.scatter(scan.x, scan.y, s=0.1, c="black")
        plt.axis("equal")
        plt.show()

    return new_scan


def clean_rotating_dipoles_by_gps(scan: HorizontalScan,
                                  raw_scan=None,
                                  max_interval_std=3,
                                  interval_length=10,
                                  min_speed=0.5):
    """
    This function uses the gps movement to estimate the rotation at each sample, then uses the
    remove_rotating_dipoles_from_rotations function to remove directional noise. It also removes samples were the
    velocity was too low or if there is a large difference between the current sample and the samples near it after the
    directional cleaning because it probably means the rotation computation was not accurate on this sample.
    Args:
        scan: HorizontalScan object.
        raw_scan: The original raw scan if one exists, it uses the original scan to compute the direction of the walk in
        each sample even if certain samples were removed.
        max_interval_std: maximum std each interval can have after directional noise cleaning, if the std is larger the
        sample is removed from the new scan.
        interval_length: length of the interval on which we compute the std.
        min_speed: the minimum speed required for the rotation to be assumed correct. In m/s.

    Returns: New scan without directional noise.
    """
    rotations, norms = get_magen_rotations(scan, raw_scan)
    correct_rotations = norms > (min_speed / 5)  # convert from m/s to samples
    scan = scan[correct_rotations]
    rotations = np.array(rotations)
    rotations = rotations[correct_rotations]
    clean_b = remove_rotating_dipoles_from_rotations(scan.b, rotations)
    good_ind = []

    for i in range(interval_length, len(scan) - interval_length - 1):
        if np.std((clean_b[i - interval_length:i + interval_length + 1])) < max_interval_std:
            good_ind.append(i)
    scan = scan[good_ind]
    rotations = rotations[good_ind]
    scan.b = remove_rotating_dipoles_from_rotations(scan.b, rotations)

    return scan


def approximate_3d_rotations_from_magen_walk(x_values, y_values, z_values):
    """
    Given near 2d-walk (specifically of the magen) on mostly the xy-plane the function estimates the rotation at each
    sample by the minimum rotation between [1,0,0] and the magen velocity.
    Args:
        x_values: x-values of the GPS.
        y_values: y-values of the GPS.
        z_values: z-values of the GPS.

    Returns: Estimated rotation at each sample and the velocities at each sample, the smaller the velocity the less
    reliable the rotation.
    """
    movements = [np.array([x_values[i + 1] - x_values[i - 1],
                           y_values[i + 1] - y_values[i - 1],
                           z_values[i + 1] - z_values[i - 1]])
                 for i in range(1, len(x_values) - 1)]
    movements = np.array([movements[0]] + movements + [movements[-1]])

    base = np.array([1., 0., 0.])
    min_rotations = [minimum_rotation_between_3d_vectors(base, movement)
                     for movement in movements]

    speeds = np.linalg.norm(movements, axis=1)

    return min_rotations, speeds


def get_magen_rotations(scan: HorizontalScan, raw_scan=None):
    """
    The following function estimates the rotation of the magen in each sample by it's GPS.

    Args:
        scan: HorizontalScan object.
        raw_scan: The original raw scan if one exists, it uses the original scan to compute the direction of the walk in
        each sample even if certain samples were removed from the new scan.

    Returns: Rotation at each sample and the velocity in each sample, the lower the velocity the less reliable the
    rotation estimation is.

    """
    if raw_scan is None:
        raw_scan = scan

    rotation_matrices, norms = approximate_3d_rotations_from_magen_walk(
        raw_scan.x,
        raw_scan.y,
        raw_scan.a,
    )

    # Nearest neighbor tree
    tree = KDTree(np.column_stack([raw_scan.x, raw_scan.y, raw_scan.a]))

    rotation_matrices = np.array([rotation_matrices[tree.query([x, y, z], 1)[1]]
                                  for x, y, z in zip(scan.x,
                                                     scan.y,
                                                     scan.a)])
    norms = np.array([norms[tree.query([x, y, z], 1)[1]]
                      for x, y, z in zip(scan.x,
                                         scan.y,
                                         scan.a)])

    return rotation_matrices, norms
