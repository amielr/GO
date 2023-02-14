import numpy as np
import pandas as pd
from pathlib import Path
from tkinter.filedialog import askopenfilename


def add_spiral_route(a, b, n=10, x0=0, y0=0, z0=10, sample_num=None):
    """
        add spiral-shaped route to scan
        :param a: shift the spiral by a samples, as in make it start a samples later
        :param b: scale factor to multiply spiral values by
        :param n: number of spirals around the center
        :param x0: the x-coord of the center
        :param y0: the y-coord of the center
        :param z0: the z-coord of the center
        :param sample_num: number of samples
        """
    # TODO add velocity and fs as parameters

    if not sample_num:
        sample_num = n * 30
    theta_array = np.linspace(0, n * np.pi, sample_num)
    r_array = a + b * theta_array

    x_array = r_array * np.cos(theta_array) + x0
    y_array = r_array * np.sin(theta_array) + y0
    z_array = np.zeros(len(x_array)) + z0

    return np.array([x_array, y_array, z_array]).T


def add_route_from_gz_file(path=None, average_around=None):
    """
    add to scan more sampling points from scan file
    :param path: file path to read route from
    :param average_around: False or tuple (x0, y0, z0)
    """
    if not path:
        path = Path(askopenfilename(filetypes=[('Text Files', ['*.txt', '*.TXT'])]))
    matrix = pd.read_csv(path, sep='\t')
    if average_around:
        matrix['x'] = matrix['x'] - matrix['x'].mean() + average_around[0]
        matrix['y'] = matrix['y'] - matrix['y'].mean() + average_around[1]
        matrix['height'] = matrix['height'] - matrix['height'].mean() + average_around[2]
    return np.array([matrix['x'], matrix['y'], matrix['height']]).T


def add_rectangular_route(x0, x1, y0, y1, z, num_lines=10, theta=0, v=5, fs=10):
    """
        add rectangular route to scan
        :param v: scanning velocity (m/s)
        :param fs: frequency of samples (Hz)
        :param num_lines: number of lines
        :param x0: initial x-coord of rectangular path
        :param x1: final x-coord of rectangular path
        :param y0: initial y-coord of rectangular path
        :param y1: final y-coord of rectangular path
        :param theta: angle to rotate the line by (radians)
        :param z: z-value of scan
        :return: np.ndarray with x,y,z coords
        """

    # calculate distance between samples and verify
    step = v / fs
    assert step > 0.00001, "The distance between samples is too small, please alter your parameters and try again."
    x_grid, y_grid = np.meshgrid(
        np.arange(x0, x1, step),
        np.linspace(y0, y1, num_lines)
    )
    mat = np.stack([x_grid, y_grid, z * np.ones(x_grid.shape)], axis=-1)
    scan_geo = mat.reshape(-1, mat.shape[-1])

    # prepare data for rotation
    mean_vector = np.array([np.mean(scan_geo[:, 0]), np.mean(scan_geo[:, 1]), np.mean(scan_geo[:, 2])])
    scan_geo = scan_geo - mean_vector

    # rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    # apply rotation + translate back to place
    scan_geo = np.matmul(R, scan_geo.T).T + mean_vector

    return scan_geo


def add_rectangular_route_by_mid_point(v, fs, x_mid, y_mid, z, width, length, theta, line_spacing):
    """

    Parameters
    ----------
    v : speed of the drone [m/s]
    fs : samplerate of the sensor [samples/second]
    x_mid : x middle point of scan
    y_mid : y middle point of scan
    z : z of scan
    width : width of polygon [m]
    length : length of polygon [m]
    theta : angle of scan long lines 0--north, 90--east in degrees
    line_spacing : the distance between each scan line in [m]

    Adds the scan route to the SimulatedScan type object
    -------

    """
    x0 = x_mid - length / 2
    x1 = x_mid + length / 2
    y0 = y_mid + width / 2
    y1 = y_mid - width / 2
    num_lines = width // line_spacing
    return add_rectangular_route(v, fs, x0, x1, y0, y1, z, num_lines=num_lines, theta=np.deg2rad(theta))
