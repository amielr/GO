import numpy as np


def direction_amplitude_decomposition(mat: np.ndarray, check_for_zeros=True):
    """
    Args:
        mat: A 2D nd-array.
        check_for_zeros: if set to False the function will assume there aren't any rows of zeros, and
        it might improve running time a bit. If the dimension of the row vectors is large it won't make any significant
        time difference but if it's small (2 or 3) it will improve running time.

    Returns:  A pair normalized_mat, norms such that,
              mat == np.array([row * norm for row, norm in zip(normalized_mat, norms)]).
    """
    norms = np.linalg.norm(mat, axis=1)

    if check_for_zeros:
        non_zero_norms = np.where(norms == 0, 1, norms)  # Replace zeros with ones.
    else:
        non_zero_norms = norms

    normalized_mat = np.divide(mat.T, non_zero_norms).T
    return normalized_mat, norms


def dot_row_pairs(mat_a: np.ndarray, mat_b: np.ndarray):
    """
    Faster version of np.array([np.dot(row_a, row_b) for row_a, row_b in zip(mat_a, mat_b))]).
    Args:
        mat_a: a 2 dimensional array.
        mat_b: a 2 dimensional array.

    Returns: one dimensional array of the dot products of each pair of rows.
    """
    return np.einsum('ij,ij->i', mat_a, mat_b)


def multiply_row_value_pairs(mat: np.ndarray, weights: np.array):
    """
    Faster version of np.array([row * weight for row, weight in zip(mat, weights))]).
    Args:
        mat: a 2 dimensional array.
        weights: a numpy array of weights to multiply each row.
    Returns: a new matrix where each row is multiplied by it's corresponding weight.
    """
    return np.einsum('ij,i->ij', mat, weights)


# subtract average from cols
def center_columns(mat: np.ndarray):
    return mat - mat.mean(axis=0)


# subtract average from rows
def center_rows(mat: np.ndarray):
    return center_columns(mat.T).T


def angle_between_vectors(vector1, vector2):
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)

    angle = 0
    if norm_1 != 0. and norm_2 != 0.:
        angle = np.arccos(vector1 @ vector2 / (norm_1 * norm_2))

    return angle


# normalize rows (divide by norm/amplitude).
def normalize_rows(mat: np.ndarray, check_for_zeros=True):
    return direction_amplitude_decomposition(mat, check_for_zeros)[0]


# normalize columns (divide by norm/amplitude).
def normalize_columns(mat: np.ndarray, check_for_zeros=True):
    return direction_amplitude_decomposition(mat.T, check_for_zeros)[0].T


def generate_orthonormal_axes(base_axis, n_axes):
    """
    Perform gram-schmidt to return orthonormal matrix with n-axes that contains the base axis.
    Args:
        base_axis: some direction vector.
        n_axes: number of desired vectors that will contain the base axis and all be orthogonal to each other.

    Returns: a nd-array that contains n_axes direction vectors that are all orthogonal to each other and contains
    the base_axis.
    """
    base_axis = base_axis / np.linalg.norm(base_axis)
    axes = [base_axis] + [np.random.randn(len(base_axis)) for _ in range(n_axes - 1)]
    q_matrix = np.linalg.qr(np.array(axes).T)[0].T
    q_matrix[0] = base_axis  # make sure the base axis was not multiplied by -1.
    return q_matrix
