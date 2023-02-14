import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
from mag_utils.scans import HorizontalScan
from pathlib import Path

B_EXT = np.array([0, 1, -1]) / np.sqrt(2) * 4.4 * 10 ** -5  # [T]
MU0 = 10 ** -7 * (4 * np.pi)


def create_scan_matrix_from_gz(scan_df, normalize_around_average=False):
    """
    :param scan_df: df of the scan with headers 'x', 'y', 'height', 'B'
    :param normalize_around_average: create scan around zero if True
    :return: array of positions of x,y,z and field in nT from real scan
    """

    scans_magnetic_field = scan_df['B'].to_numpy() * 10 ** -9
    scan_pos_mat = scan_df[['x', 'y', 'height']].to_numpy()
    if normalize_around_average:
        # create scan positions around (0,0,0) axis
        scan_pos_mat = scan_pos_mat - scan_pos_mat.mean(axis=0)
    return scans_magnetic_field, scan_pos_mat


def create_scan_matrix_from_mag_scan(scan: HorizontalScan, normalize_around_average=False):
    """
    :param scan: Horizontal mag Scan object that contains the route
    :param normalize_around_average: create scan around zero if True
    :return: array of positions of x,y,z and field in nT from real scan
    """

    scans_magnetic_field = scan.b * 10 ** -9
    scan_pos_mat = np.column_stack([scan.x, scan.y, scan.a])
    if normalize_around_average:
        # create scan positions around (0,0,0) axis
        scan_pos_mat = scan_pos_mat - scan_pos_mat.mean(axis=0)
    return scans_magnetic_field, scan_pos_mat


def distance_from_predicted_source(source_pos, scan_pos_mat):
    """
    :source_pos: array of positions of source [x,y,z]
    :scan_pos_mat: array (3,N) of [x, y, z] of scan
    :returns: array (3,N) of distance from source

    """
    return scan_pos_mat - source_pos


def magnetic_dipole_formula(source_pos, source_moment, scan_pos):
    """

    """
    r = scan_pos[np.newaxis, :, :] - source_pos[:, np.newaxis, :]
    r_norms = np.linalg.norm(r, axis=-1)
    r_normalized = np.einsum('ijk,ij->ijk', r, 1 / r_norms)
    dist_moment_dot_prod = np.einsum('ijk,ik->ij', r_normalized, source_moment)

    first_part = 3 * np.einsum('ij, ijk -> ijk', dist_moment_dot_prod, r_normalized)  # 3 * (r dot m) * r
    magnetic_field_vector = (MU0 / (4 * np.pi)) * \
                            (first_part.transpose(1, 0, 2) - source_moment).transpose(2, 1, 0) / (r_norms ** 3)
    return magnetic_field_vector.transpose(1, 2, 0)  # particle, sample, Bxyz


def create_magnetic_dipole_simulation(source_pos, source_moment, scan_pos_mat, B_ext=B_EXT, field_per_dipole=False,
                                      scalar=True):
    """
    :param source_pos: array of [[x1,y1,z1],[x2,y2,z2], ....] of source
    :param source_moment: array of [[mx, my, mz]] of all sources or
                          [[mx0, my0, mz0], [mx1, my1, mz1], ... , [mxn, myn, mzn]] for each source
    :param scan_pos_mat: array of positions of x,y,z from real scan
    :return: scan like array of B simulation (magnetic field) given parameters
    """

    if source_pos.shape != source_moment.shape:
        source_moment = np.array([source_moment[0]] * len(source_pos))

    # calculate B for each source
    B = magnetic_dipole_formula(source_pos, source_moment, scan_pos_mat)

    if scalar:
        if field_per_dipole:
            magnetic_field = B + B_ext
            return np.linalg.norm(magnetic_field, axis=2)
        else:
            B = B.sum(axis=0)  # B = sum of magnetic fields created by all dipoles
            magnetic_field = B + B_ext
            return np.linalg.norm(magnetic_field, axis=1)
    else:
        if field_per_dipole:
            magnetic_field = B + B_ext
            return magnetic_field
        else:
            B = B.sum(axis=0)  # B = sum of magnetic fields created by all dipoles
            magnetic_field = B + B_ext
            return magnetic_field


if __name__ == "__main__":
    path = Path(
        r"<insert path here>")
    real_scan_df = pd.read_csv(path, delimiter='\t')
    # B_scan, scan_route_mat = create_scan_matrix_from_gz(real_scan_df, normalize_around_average=True)
    source_positions = np.array([[0, 0, 0]])
    source_moments = np.array([[0, 10, -10]])
    scan_route_mat = np.array([[10, 10, 5]])
    B_simu = create_magnetic_dipole_simulation(source_positions, source_moments, scan_route_mat)

    plt.tricontourf(scan_route_mat[:, 0], scan_route_mat[:, 1], B_simu, levels=100)
    # plt.tricontour(scan_route_mat[:, 0], scan_route_mat[:, 1], B_simu, colors='black', linewidths=0.2, levels=10)
    plt.show()
