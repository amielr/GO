import numpy as np
from numpy.linalg import norm
from sklearn.cluster import DBSCAN
from mag_utils.mag_utils.algo_tools.np_wrappers import direction_amplitude_decomposition

# set consts
ROTATION_MATRIX = np.array([[0, -1],
                            [1, 0]])  # 90 degrees rotation.


def angle_between_vects(vec1: np.array, vec2: np.array, in_radians=False):
    """
    Args:
        vec1: A d-dimensional vector.
        vec2: A d-dimensional vector.
        in_radians: If True angle will be returned in radians, otherwise, in degrees.

    Returns: The angle between vec1 and vec2.
    """

    # compute the cosine of the angle between the vectors using dot product.
    # ----------------------------------------------------------------------
    dot_vects = vec1 @ vec2
    norm1 = norm(vec1)
    norm2 = norm(vec2)

    if norm1 > 0 and norm2 > 0:
        dot_vects /= norm1 * norm2
    # ----------------------------------------------------------------------

    # compute the angles in degrees / radians:
    # ----------------------------------------
    angle = np.arccos(dot_vects)

    if not in_radians:
        angle *= 180 / np.pi
    # ----------------------------------------

    return angle


def split_to_opposite_dirs(dirs_2d,
                           normalize=False,
                           em_iterations=1,
                           major_minor_dirs=None):
    """
    The function finds the direction most vectors are on and returns which vectors are looking in one direction of it,
    which vectors in the opposite direction and which vectors are more orthogonal to it.
    Args:
        dirs_2d: An nd-array of n 2-dimensional vectors.
        normalize: If set to True all the vectors are normalized (and has equal weights). Otherwise a vector which is
        twice has large influences twice as much.
        em_iterations: In general you can ignore this parameter, its goal is to deal with certain rare edge cases.
        If em_iterations > 0 the function calls itself recursively but removes vectors that are
        orthogonal to the direction we computed for the lines so they won't influnce the major direction of the lines.
        major_minor_dirs: Ignore this parameter, it's used only for the recursion in the em_iterations.
    Returns: 3 boolean arrays that represent the indices of the points that are in one direction the ones that are on
    the opposite directions and the ones that are orthogonal to the main direction.
    """
    normalized_dirs, _ = direction_amplitude_decomposition(dirs_2d)

    if normalize:
        dirs_2d = normalized_dirs
    if major_minor_dirs is None:
        _, _, major_minor_dirs = np.linalg.svd(dirs_2d)

    major = normalized_dirs @ major_minor_dirs[0]
    minor = normalized_dirs @ major_minor_dirs[1]

    major[np.abs(major) < np.abs(minor)] = 0

    if em_iterations > 0:
        major_minor_dirs = np.linalg.svd(dirs_2d[major != 0])[-1]
        return split_to_opposite_dirs(dirs_2d,
                                      em_iterations=em_iterations - 1,
                                      major_minor_dirs=major_minor_dirs)

    return major > 0, major < 0, major == 0


def cluster_lines_by_density(scan, cluster_distance, min_cluster_size):
    """
    Args:
        scan: mag scan object with the lines to be clustered.
        cluster_distance: maximum distance (in meters) between adjacent points to be considered in the same cluster.
        min_cluster_size: minimum number of samples in a cluster to be considered a "real" cluster.
    Returns:
        An array of the clusters of each point.
    """
    dbscan_model = DBSCAN(eps=cluster_distance,
                          min_samples=min_cluster_size)
    return dbscan_model.fit(list(zip(scan.x, scan.y))).labels_


def cluster_lines_by_direction(scan, line_segments_indices, clusters):
    """
    Args:
        scan: mag scan.
        line_segments_indices: a list of lists of the indices of segments that are on the same line.
        clusters: DB-scan clusters of the points, points that are not in the same cluster won't be defined to be in the
        same line regardless of their direction.

    Returns: A list of lists of the indices of each line.
    """
    lines_indices = [line_segments_indices[0]]

    for line_ind in line_segments_indices[1:]:
        # start point of previous line
        prev_start = np.array([scan.x[lines_indices[-1][0]], scan.y[lines_indices[-1][0]]])

        # end point of previous line
        prev_end = np.array([scan.x[lines_indices[-1][-1]], scan.y[lines_indices[-1][-1]]])

        # start point of current line
        current_start = np.array(
            [scan.x[line_ind[0]], scan.y[line_ind[0]]])

        # end point of current line
        current_end = np.array(
            [scan.x[line_ind[-1]], scan.y[line_ind[-1]]])

        prev_line_segment_vector = prev_end - prev_start
        current_line_segment_vector = current_end - current_start
        sum_size = norm(prev_line_segment_vector + current_line_segment_vector)

        # if the two line segments are in the same direction and cluster merge them, otherwise start new line.
        if clusters[line_ind[0]] == clusters[lines_indices[-1][0]] \
                and sum_size > norm(prev_line_segment_vector) \
                and sum_size > norm(current_line_segment_vector):
            lines_indices[-1].extend(line_ind)
        else:
            lines_indices.append(line_ind)
    return lines_indices


def preliminary_clustering(scan, cluster_distance, min_cluster_size, time_resolution):
    """
    Args:
        scan: mag scan.
        cluster_distance: maximum distance between adjecent points to be considered different lines regardless of the
        direction.
        min_cluster_size: minimum number of points in a cluster to not be removed.
        time_resolution: time resolution for line segments to cluster.

    Returns: a list of the indices to remove from the data and a list of lists of the indices of each line.

    """
    # perform dbscan to separate the lines initially
    clusters = cluster_lines_by_density(scan=scan,
                                        cluster_distance=cluster_distance,
                                        min_cluster_size=min_cluster_size)

    indices_to_remove = set(np.where(clusters == -1)[0])
    lines_indices_all = np.where(clusters != -1)[0]  # all the points that were not
    # removed

    line_segments_indices = []
    prev_cluster = -1

    # map out line segments based on dbscan results
    for j, i in enumerate(lines_indices_all):
        current_cluster = clusters[i]
        if j % int(time_resolution * 10) == 0 or current_cluster != prev_cluster:
            line_segments_indices.append([])
        line_segments_indices[-1].append(i)
        prev_cluster = current_cluster

    # improve clustering using direction of lines
    line_segments_indices = cluster_lines_by_direction(scan=scan,
                                                       line_segments_indices=line_segments_indices,
                                                       clusters=clusters)
    sizes = [
        norm([scan.x[line[-1]] - scan.x[line[0]],
              scan.y[line[-1]] - scan.y[line[0]]])
        for line in line_segments_indices]

    # calculate median
    median = np.median(sizes)

    # check if line's length is too small to be real and if it is, remove
    lines_indices_fixed = [line if size >= median / 2 else indices_to_remove.update(set(line)) for line, size in
                           zip(line_segments_indices, sizes)]
    lines_indices_fixed = [i for i in lines_indices_fixed if i is not None]

    return indices_to_remove, lines_indices_fixed


def get_line_direction(scan, line_indices):
    """
    Args:
        scan: scan
        line_indices: indices of a line.

    Returns: A vector of the direction of the line, which is defined to be the vector between the start and end of the
    middle half of the points (because the edges are sometimes not really in the same direction).
    """
    point_towards_end = np.array([scan.x[line_indices[3 * len(line_indices) // 4]],
                                  scan.y[line_indices[3 * len(line_indices) // 4]]])
    point_towards_start = np.array([scan.x[line_indices[len(line_indices) // 4]],
                                    scan.y[line_indices[len(line_indices) // 4]]])

    return point_towards_end - point_towards_start


def get_line_segments_indices(scan,
                              time_resolution,
                              perpendicular_state_flag,
                              angle_tolerance_value,
                              opposite_state_flag,
                              cluster_distance, min_cluster_size):
    """
    Args:
        scan: mag_scan object
        time_resolution: the window size to look at when calculating the direction of movement (in seconds).
        perpendicular_state_flag: If set to true removes points that are perpendicular to their line.
        angle_tolerance_value: The angle tolerance is the maximum angle of points to their lines
        at which we remove the points (in degrees)
        opposite_state_flag: If set to true removes points that walked backwards from their line.
        cluster_distance: The cluster distance is the minimum distance needed for samples to be considered in the same
        line.
        min_cluster_size: minimum number of samples in a cluster

    Returns: A list of lists of the indices of the lines.
    """
    # perform preliminary clustering
    indices_to_remove, lines_indices = preliminary_clustering(scan,
                                                              cluster_distance,
                                                              min_cluster_size,
                                                              time_resolution)

    # separate to lines using logic
    for i, line_ind in enumerate(lines_indices):
        line_dir = get_line_direction(scan, line_ind)

        # add to indices to be removed
        indices_to_remove.add(line_ind[-1])
        indices_to_remove.add(line_ind[0])
        for j in range(len(line_ind) - 1):
            # direction to next line
            dir_to_next = np.array([scan.x[line_ind[j + 1]], scan.y[line_ind[j + 1]]]) - np.array([
                scan.x[line_ind[j]], scan.y[line_ind[j]]])
            angle_to_rotated = min(angle_between_vects(dir_to_next, ROTATION_MATRIX @ line_dir),
                                   angle_between_vects(dir_to_next, -ROTATION_MATRIX @ line_dir))

            # perform cutting according to lines
            if perpendicular_state_flag and angle_to_rotated < angle_tolerance_value or \
                    opposite_state_flag and norm(dir_to_next + line_dir) < norm(line_dir):
                indices_to_remove.add(line_ind[j])

        # remove the indices in the indices to remove from thhe line indices.
        lines_indices[i] = [x for x in line_ind if x not in indices_to_remove]

    return lines_indices


def substract_mean_by_dir(lines_indices, scan):
    """
    Args:
        lines_indices: the line index corresponding to each point in the scan.
        scan: a magScan object.

    Returns: magScan object that contains B values with mean substracted by direction.

    """
    line_indices = np.array(lines_indices)
    line_dirs = np.array([np.array([scan.x[line_ind[0]] - scan.x[line_ind[-1]],
                                    scan.y[line_ind[0]] - scan.y[line_ind[-1]]])
                          for line_ind in
                          line_indices])
    major_1, major_2, _ = split_to_opposite_dirs(line_dirs)
    m_1 = np.mean(np.concatenate([scan.b[line_ind]
                                  for line_ind in line_indices[major_1]]).ravel())
    m_2 = np.mean(np.concatenate([scan.b[line_ind]
                                  for line_ind in line_indices[major_2]]).ravel())

    for line_ind in line_indices[major_1]:
        scan.b[line_ind] -= m_1
    for line_ind in line_indices[major_2]:
        scan.b[line_ind] -= m_2

    return scan


def substract_mean_per_line(lines_indices, scan):
    """
    Args:
        lines_indices: a list of lists of the indices of each line.
        scan: mag scan.

    Returns: subtract the mean B value from each line and returns the scan.
    """
    # subtracting each line's mean if checked
    for line_ind in lines_indices:
        scan.b[line_ind] -= scan.b[line_ind].mean()

    return scan


def separate_lines(scan,
                   cluster_distance,
                   min_cluster_size,
                   time_resolution,
                   perpendicular_state_flag,
                   angle_tolerance_value,
                   opposite_state_flag,
                   mean_by_dir_flag,
                   substract_mean_per_line_flag):
    """
    Args:
        scan: magScan object
        time_resolution: the window size to look at when calculating the direction of movement (in seconds).
        perpendicular_state_flag: If set to true removes points that are perpendicular to their line.
        angle_tolerance_value: The angle tolerance is the maximum angle of points to their lines
        at which we remove the points (in degrees)
        opposite_state_flag: If set to true removes points that walked backwards from their line.
        cluster_distance: The cluster distance is the minimum distance needed for samples to be considered in the same
        line.
        min_cluster_size: minimum number of samples in a cluster
        mean_by_dir_flag: If mean should be substracted per direction, this should be true.
        substract_mean_per_line_flag: If mean should be substracted per line, this should be true.

    Returns: magScan with lines separated, Line indices corresponding to each point in scan.

    """
    lines_indices_fixed = get_line_segments_indices(scan, time_resolution, perpendicular_state_flag,
                                                    angle_tolerance_value, opposite_state_flag, cluster_distance,
                                                    min_cluster_size)

    if mean_by_dir_flag:
        scan = substract_mean_by_dir(lines_indices_fixed, scan)

    if substract_mean_per_line_flag:
        scan = substract_mean_per_line(lines_indices_fixed, scan)

    return scan, lines_indices_fixed
