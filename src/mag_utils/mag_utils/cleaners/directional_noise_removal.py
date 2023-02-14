import numpy as np
from mag_utils.algo_tools.np_wrappers import angle_between_vectors


def remove_rotating_dipoles_from_rotations(b, rotation_matrices):
    """
    Given the rotation matrices at each sample of the scan, the following funciton removes
    the influence of any rotating dipole that rotates with the sensor. The noise model created by the rotating dipoles
    can be expressed by a linear combination of the 9 numbers in the rotation matrix and a constant, so by multiplying
    the scan values by the rejection matrix of the span created by those 10 variables we get a clean scan.
    Args:
        b: Magnetic field amplitude measurements.
        rotation_matrices: The rotation in each sample of the sensor.

    Returns: New b values without directional noise.

    """
    noise_mat = np.array([rotation.ravel() for rotation in rotation_matrices])
    noise_mat = np.column_stack([np.ones_like(b), noise_mat])
    noise_mat = np.linalg.qr(noise_mat)[0]
    b = b - noise_mat @ (noise_mat.T @ b)

    return b


def minimum_rotation_between_3d_vectors(base_vector, new_vector):
    """
    Given 2 3d vectors base_vector, new_vector the function computes which rotation changed base_vector to new_vector,
    but since a rotatation around base_vector cannot be detected, there are infinite such rotations, but we assume the
    rotation is only around the axis which is orthogonal to both base_vector and new_vector.
    Args:
        base_vector: Vector before the rotation.
        new_vector: Vector after the rotation.

    Returns: (3,3) minimum rotation matrix that transforms base_vector to new_vector.
    """
    if np.linalg.norm(new_vector) < 1e-10:
        return np.identity(3)

    base_vector = np.array(base_vector, dtype=float)
    new_vector = np.array(new_vector, dtype=float)

    rotation_axis = np.cross(base_vector, new_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    theta = angle_between_vectors(base_vector, new_vector)
    cross_mat = np.cross(rotation_axis, -np.identity(3))

    minimum_rotation_matrix = np.identity(3) + cross_mat * np.sin(theta) + cross_mat @ cross_mat * (1 - np.cos(theta))

    return minimum_rotation_matrix
