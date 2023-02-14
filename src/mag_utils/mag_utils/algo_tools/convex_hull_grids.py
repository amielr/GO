import numpy as np
from scipy.spatial import Delaunay


def in_hull(points, hull_points):
    """
    Args:
        points: an nd-array of points
        hull_points: an nd-array of points

    Returns: A boolean array that returns True for any point in points that is inside the convex hull of the points in
    hull_points and False for any point in points that is outside the convex hull.

    """
    return Delaunay(hull_points).find_simplex(points) >= 0


def convex_hull_points(x, y):
    """
    Args:
        x: The x-values of 2d points
        y: The y-values of 2d points

    Returns: The subset of points that defines the convex hull of the points (the points in the edges).
    """
    # pylint: disable=unsubscriptable-object
    points = np.array([x, y]).T
    delauny_hull_indexes = list(set(Delaunay(points).convex_hull.ravel()))  # noqa: R0902
    return points[delauny_hull_indexes]


def convex_hull_grid(x: np.array, y: np.array, spacing: float = 1, convex_hull_factor: float = 1):
    """
    Args:
        x: The x values of 2D points.
        y: The y values of 2D points.
        spacing:  Spacing of the grid that will be created inside the convex hull.
        convex_hull_factor: stretches the distance between every point to in the grid to the mean of the grid.
        If convex_hull_factor > 1 there will be points outside of the convex hull, and if convex_hull_factor < 1 there
        will be no points near the edge of the convex hull. To preserve the spacing, spacing will be changed to:
        spacing = spacing / convex_hull_factor.

    Returns: x, y coordinates of the grid points created inside the convex hull (or outside if convex_hull_factor > 1).
    """
    spacing /= convex_hull_factor
    x_range = np.arange(x.min(), x.max(), spacing)
    y_range = np.arange(y.min(), y.max(), spacing)
    grid_x = np.array(list(x_range) * len(y_range))
    grid_y = np.concatenate(np.array([[y] * len(x_range) for y in y_range]))

    points_in_hull = in_hull(list(zip(grid_x, grid_y)), np.array([x, y]).T)
    x_vector = grid_x[points_in_hull]
    y_vector = grid_y[points_in_hull]

    # Strech grid according to the convex hull factor.
    if convex_hull_factor != 1:
        x_mean = x_vector.mean()
        y_mean = y_vector.mean()
        x_vector = (x_vector - x_mean) * convex_hull_factor + x_mean
        y_vector = (y_vector - y_mean) * convex_hull_factor + y_mean

    return x_vector, y_vector


# create a grid based on existing scan coordinates in (x,y) convex hull
def convex_hull_grid_at_depths(x, y, depths, spacing: float = 1, convex_hull_factor: float = 1):
    """
    Args:
        x: The x values of 2D points.
        y: The y values of 2D points.
        depths: depths at which the points should be placed.
        spacing:  Spacing of the grid that will be created inside the convex hull.
        convex_hull_factor: stretches the distance between every point to in the grid to the mean of the grid.
        If convex_hull_factor > 1 there will be points outside of the convex hull, and if convex_hull_factor < 1 there
        will be no points near the edge of the convex hull. To preserve the spacing, spacing will be changed to:
        spacing = spacing / convex_hull_factor.

    Returns: x, y, z coordinates of the grid points created inside the convex hull.
    """
    x_vector, y_vector = convex_hull_grid(x, y, spacing=spacing, convex_hull_factor=convex_hull_factor)
    z_vector = np.array([[z] * len(x_vector) for z in depths]).ravel()
    x_vector = np.tile(x_vector, len(depths))
    y_vector = np.tile(y_vector, len(depths))

    return x_vector, y_vector, z_vector


def bounding_rectangle_params(points: np.ndarray, alpha):
    """
    Args:
        points: 2D points.
        alpha: angle of one of the sides of the rectangle.

    Returns: The minimum area rectangle that encloses all the points and is tilted by alpha radians.
    The function returns the center of the rectangle the side lengths and the sides'
     directions (unit direction vectors).
    """
    # Direction vectors of the sides.
    vector_1 = np.array([np.cos(alpha),
                         np.sin(alpha)])
    vector_2 = np.array([np.cos(alpha + np.pi / 2),
                         np.sin(alpha + np.pi / 2)])

    # Project the points onto the direction vectors.
    proj_v1 = points @ vector_1
    proj_v2 = points @ vector_2

    # find the center of the rectangle:
    # ---------------------------------------------------
    v1_center = (proj_v1.min() + proj_v1.max()) / 2
    v2_center = (proj_v2.min() + proj_v2.max()) / 2

    center = vector_1 * v1_center + vector_2 * v2_center
    # ---------------------------------------------------

    # find the side lengths of the rectangle and set the major
    # to be the larger one and the minor to be the smaller one.
    # -------------------------------------------------------------
    major = (proj_v1.max() - proj_v1.min())  # first side length.
    minor = (proj_v2.max() - proj_v2.min())  # second side length.

    if major < minor:
        minor, major = major, minor
        major_dir, minor_dir = vector_2, vector_1
    else:
        major_dir, minor_dir = vector_1, vector_2
    # -------------------------------------------------------------

    return center, major, minor, major_dir, minor_dir


def minimum_enclosing_rectangle(x: np.array, y: np.array, n_angles=100):
    """
       Args:
            x: The x values of 2D points.
            y: The y values of 2D points.
            n_angles: number of angles for the rotation of the rectangle to try in the optimization.

       Returns: The minimum area rectangle that encloses all the points. The function returns the center of the
       rectangle the side lengths and the sides' directions (unit direction vectors).
    """
    best_area = np.inf
    points = convex_hull_points(x, y)

    best_rectangle_params = None  # (center, major, minor, maj_dir, min_dir)

    for alpha in np.linspace(0, np.pi / 2, n_angles):
        params = bounding_rectangle_params(points, alpha)  # params: a Tuple of (center, major, minor, maj_dir, min_dir)

        area = params[1] * params[2]  # minor * major
        if area < best_area:
            best_rectangle_params = params
            best_area = area

    return best_rectangle_params


def rectangular_convex_hull_grid(x, y, spacing=1., convex_hull_factor=1.):
    """
    Args:
        x: The x values of 2D points.
        y: The y values of 2D points.
        spacing:  Spacing of the grid that will be created inside the convex hull.
        convex_hull_factor: stretches the distance between every point to in the grid to the mean of the grid.
        If convex_hull_factor > 1 there will be points outside of the convex hull, and if convex_hull_factor < 1 there
        will be no points near the edge of the convex hull. To preserve the spacing, spacing will be changed to:
        spacing = spacing / convex_hull_factor.

    Returns: x, y coordinates of the RECTANGULAR grid of points which encloses the convex hull,
    the number of points in the major and minor sides of the rectangular grid.
    """
    center, major, minor, major_dir, minor_dir = minimum_enclosing_rectangle(x, y)

    major *= convex_hull_factor
    minor *= convex_hull_factor

    n_major = int(major // spacing + 1)
    n_minor = int(minor // spacing + 1)

    x_vector, y_vector = np.meshgrid(np.linspace(-major / 2, major / 2, n_major),
                                     np.linspace(-minor / 2, minor / 2, n_minor))

    rectangle = np.array([a * major_dir + b * minor_dir for a, b in zip(x_vector.ravel(),
                                                                        y_vector.ravel())])
    rectangle += center

    return rectangle[:, 0], rectangle[:, 1], n_major, n_minor


def rectangular_convex_hull_grid_at_depths(x, y, depths, spacing: float = 1, convex_hull_factor: float = 1):
    """
    Args:
        x: The x values of 2D points.
        y: The y values of 2D points.
        depths: depths at which the points should be placed.
        spacing:  Spacing of the grid that will be created inside the convex hull.
        convex_hull_factor: stretches the distance between every point to in the grid to the mean of the grid.
        If convex_hull_factor > 1 there will be points outside of the convex hull, and if convex_hull_factor < 1 there
        will be no points near the edge of the convex hull. To preserve the spacing, spacing will be changed to:
        spacing = spacing / convex_hull_factor.

    Returns: x, y, z coordinates of the RECTANGULAR grid of points which encloses the convex hull,
    the number of points in the major and minor sides of the rectangular grid.
    """
    x_vector, y_vector, n_major, n_minor = rectangular_convex_hull_grid(x, y,
                                                                        spacing=spacing,
                                                                        convex_hull_factor=convex_hull_factor)
    z_vector = np.array([[z] * len(x_vector) for z in depths]).ravel()
    x_vector = np.tile(x_vector, len(depths))
    y_vector = np.tile(y_vector, len(depths))
    return x_vector, y_vector, z_vector, n_major, n_minor
