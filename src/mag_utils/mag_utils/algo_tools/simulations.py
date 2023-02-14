from numpy.linalg import norm
import numpy as np

from mag_utils.algo_tools.np_wrappers import direction_amplitude_decomposition, dot_row_pairs


# For scalar sensors set b_ax to be the direction of Earth's magnetic field.
def simulate_b_ax_from_dipole(dipole_pos, dipole_moment, b_ax, scan_xyz, remove_dc=False):
    """
    Args:
        dipole_pos: x, y, z position of the dipole to simulate
        dipole_moment: magnetic moment of the dipole to simulate
        b_ax: The axis at which the sensor measures the magnetic field, a scalar sensor that measures the amplplitude of
        the magnetic field can be approximated by setting b_ax=np.array([0., 1., -1.]) and adding |B_e| to the result.
        scan_xyz: an nd-array of shape (n,3) of the x,y,z posisions of the sensor during it's measurements.
        remove_dc: If true subtract the mean from the simulation.

    Returns: A vector of length n of what the sensor should have seen if there was a dipole at dipole_pos with a moment
    of dipole_moment.
    """

    b_ax = b_ax / norm(b_ax)
    moment_size = norm(dipole_moment)

    radii = np.asarray(scan_xyz) - dipole_pos  # The vectors between the dipole and the scan positions.
    r_normalized, r_norms = direction_amplitude_decomposition(radii, check_for_zeros=False)

    if norm(b_ax - (dipole_moment / moment_size)) < 1e-10:
        # if the moment is in the same direction as b_ax we can make the computation faster.
        b = ((r_normalized @ (np.sqrt(3 * moment_size) * b_ax)) ** 2 - moment_size) / (r_norms ** 3)
    else:
        b = ((r_normalized @ (3 * b_ax)) * (r_normalized @ dipole_moment) - b_ax @ dipole_moment) / (r_norms ** 3)

    if remove_dc:
        b -= b.mean()

    return b


def simulate_b_ax_from_xyz_moments(dipole_pos, b_ax, scan_xyz, remove_dc=False):
    """
       Args:
           dipole_pos: x, y, z position of the dipole to simulate
           b_ax: The axis at which the sensor measures the magnetic field, a scalar sensor that measures the amplitude
            of the magnetic field can be approximated by setting b_ax=np.array([0., 1., -1.]) and adding |B_e|
            to the result.
           scan_xyz: an nd-array of shape (n,3) of the x,y,z posisions of the sensor during it's measurements.
           remove_dc: If true subtract the mean from the simulation.

       Returns: 3 simulation vectors, one that shows what the sensor should have seen if theres was a dipole at
       dipole_pos with a magnetic moment of [1, 0, 0] one with a moment [0, 1, 0] and one with a moment of [0, 0, 1].

        For scalar sensors set b_ax to be the direction of Earth's magnetic field.
       This function is faster than calling the simulate_b_ax_from_dipole 3 time.
       """
    b_ax = b_ax / norm(b_ax)

    radii = np.asarray(scan_xyz) - dipole_pos  # The vectors between the dipole and the scan positions.
    r_normalized, r_norms = direction_amplitude_decomposition(radii, check_for_zeros=False)

    r_inv_cubed = r_norms ** -3
    r_proj_b_times_3 = (r_normalized @ (3 * b_ax))

    b_from_x = (r_proj_b_times_3 * r_normalized[:, 0] - b_ax[0]) * r_inv_cubed  # noqa: R0902
    b_from_y = (r_proj_b_times_3 * r_normalized[:, 1] - b_ax[1]) * r_inv_cubed  # noqa: R0902
    b_from_z = (r_proj_b_times_3 * r_normalized[:, 2] - b_ax[2]) * r_inv_cubed  # noqa: R0902

    if remove_dc:
        b_from_x -= b_from_x.mean()
        b_from_y -= b_from_y.mean()
        b_from_z -= b_from_z.mean()

    return b_from_x, b_from_y, b_from_z


def simulate_b_ax_from_each_dipole(dipoles_pos, dipoles_moments, b_ax, scan_pos):
    """
       Args:
           dipoles_pos: A (3,n) array of the x, y, z positions of the dipoles to simulate.
           b_ax: The axis at which the sensor measures the magnetic field, a scalar sensor that measures the amplitude
           of the magnetic field can be approximated by setting b_ax=np.array([0., 1., -1.]) and adding |B_e| to
            the result.
           scan_pos: The x,y,z position of the sensor.

       Returns: An np.array of the simulation of each dipole at the sensors location.
       """
    b_ax = b_ax / norm(b_ax)  # normalize axis
    radii = np.asarray(dipoles_pos) - scan_pos  # The vectors between the dipoles and the scan positions.
    r_normalized, r_norms = direction_amplitude_decomposition(radii, check_for_zeros=False)
    r_proj_3ax = r_normalized @ (3 * b_ax)
    return dot_row_pairs((r_normalized.T * r_proj_3ax).T - b_ax, dipoles_moments) / r_norms ** 3


def find_moments_with_orthogonal_simulations(dipole_pos, b_ax, scan_positions, orthonormalize=True,
                                             remove_dc=False):
    """
    The function computes 3 possible moments m1, m2, m3 and three corresponding simulations (simulation of the magnetic
    field created by the dipoles at scan positions) s1, s2, s3 such that s1, s2 and s3 are orthogonal to each other.
    Those 3 simulations span the subspace of all the simulations that can be created by a dipole at dipole_pos.
    If orthonormalize=True the cross-correlation matrix of the 3 simulations is the projection matrix of that subspace
    (i.e., norm(s1) = norm(s2) = norm(s3) = 1).
    If orthonormalize=False then norm(m1) = norm(m2) = norm(m3) and m1 = base_moment.
    For further explanation talk to Tomer Wolberg.
    Args:
        dipole_pos: position of dipole in (x,y,z) form
        b_ax: the axis along which the magnetic field is calculated.
        scan_positions: the location of scan samples
        orthonormalize: boolean value, If True, the returned simulations has norm 1 if False the returned moments has
        norm of norm(base_moment).
        remove_dc: a boolean that tells whether or not to remove the const in the simulations.

    Returns: m1, m2, m3, s1, s2, s3.
    """

    moments = np.identity(3)

    simulations = simulate_b_ax_from_xyz_moments(dipole_pos, b_ax, scan_positions,
                                                 remove_dc=remove_dc)
    simulations = list(simulations)
    base_simulation_norm = norm(simulations[0])

    if orthonormalize:
        moments[0] = moments[0] / base_simulation_norm
        simulations[0] /= base_simulation_norm
        base_simulation_norm = 1

    a_proj_base_part = (simulations[1] @ simulations[0]) / base_simulation_norm ** 2
    moments[1] -= a_proj_base_part * moments[0]
    simulations[1] -= a_proj_base_part * simulations[0]
    sim_a_norm = norm(simulations[1])
    simulations[1] *= base_simulation_norm / sim_a_norm
    moments[1] *= base_simulation_norm / sim_a_norm

    b_proj_base_part = (simulations[2] @ simulations[0]) / base_simulation_norm ** 2
    b_proj_a_part = (simulations[2] @ simulations[1]) / base_simulation_norm ** 2
    moments[2] -= b_proj_base_part * moments[0]
    simulations[2] -= b_proj_base_part * simulations[0]
    moments[2] -= b_proj_a_part * moments[1]
    simulations[2] -= b_proj_a_part * simulations[1]

    sim_b_norm = norm(simulations[2])
    simulations[2] *= base_simulation_norm / sim_b_norm
    moments[2] *= base_simulation_norm / sim_b_norm

    return moments[0], moments[1], moments[2], simulations[0], simulations[1], simulations[2]
