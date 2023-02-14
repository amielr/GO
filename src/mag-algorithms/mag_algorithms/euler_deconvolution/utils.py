"""
Util functions for use inside the algorithms themselves.

This file also contains functions we copied and slightly changed from the fatiando library

.. note:: Most, if not all, fatiando functions here require gridded data.
**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of
  gridded potential field data on a level surface.
* :func:`~fatiando.gravmag.transform.reduce_to_pole`: Reduce the total field
  magnetic anomaly to the pole.
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)
* :func:`~fatiando.gravmag.transform.tilt`: Calculates the tilt angle
* :func:`~fatiando.gravmag.transform.power_density_spectra`: Calculates
  the Power Density Spectra of a gridded potential field data.
* :func:`~fatiando.gravmag.transform.radial_average`: Calculates the
  the radial average of a Power Density Spectra using concentring rings.

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction (North-South)
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction (East-West)
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

"""
from __future__ import division
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def derivx(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the x direction.

    Warning:
        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Args:
        x: the x coordinates of the grid points
        y: the y coordinates of the grid points
        data: the potential field at the grid points
        shape: the shape of the grid
        order: the order of the derivative
        method: the method used to calculate the derivatives. Options are:
        'fd' for central finite-differences (more stable) or 'fft' for
        the fast fourier transform

    Returns:
        deriv: the derivative in the x axis
    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        kx, _ = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = np.fft.fft2(padded) * (kx * 1j) ** order
        deriv_pad = np.real(np.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dx = (x.max() - x.min()) / (nx - 1)
        deriv = np.empty_like(datamat)
        deriv[1:-1, :] = (datamat[2:, :] - datamat[:-2, :]) / (2 * dx)
        deriv[0, :] = deriv[1, :]
        deriv[-1, :] = deriv[-2, :]
        if order > 1:
            deriv = derivx(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivy(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the x direction.

    Warning:
        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Args:
        x: the x coordinates of the grid points
        y: the y coordinates of the grid points
        data: the potential field at the grid points
        shape: the shape of the grid
        order: the order of the derivative
        method: the method used to calculate the derivatives. Options are:
        'fd' for central finite-differences (more stable) or 'fft' for
        the fast fourier transform

    Returns:
        deriv: the derivative in the y axis
    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        _, ky = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = np.fft.fft2(padded) * (ky * 1j) ** order
        deriv_pad = np.real(np.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dy = (y.max() - y.min()) / (ny - 1)
        deriv = np.empty_like(datamat)
        deriv[:, 1:-1] = (datamat[:, 2:] - datamat[:, :-2]) / (2 * dy)
        deriv[:, 0] = deriv[:, 1]
        deriv[:, -1] = deriv[:, -2]
        if order > 1:
            deriv = derivy(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivz(x, y, data, shape, order=1, method='fft'):
    """
    Calculate the derivative of a potential field in the x direction.

    Warning:
        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Args:
        x: the x coordinates of the grid points
        y: the y coordinates of the grid points
        data: the potential field at the grid points
        shape: the shape of the grid
        order: the order of the derivative
        method: the method used to calculate the derivatives. Options are:
        'fd' for central finite-differences (more stable) or 'fft' for
        the fast fourier transform

    Returns:
        deriv: the derivative in the z axis
    """
    assert method == 'fft', \
        "Invalid method '{}'".format(method)
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    deriv_ft = np.fft.fft2(padded) * np.sqrt(kx ** 2 + ky ** 2) ** order
    deriv = np.real(np.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx: padx + nx, pady: pady + ny].ravel()


def _pad_data(data, shape):
    n = _nextpow2(np.max(shape))
    nx, ny = shape
    padx = (n - nx) // 2
    pady = (n - ny) // 2
    padded = np.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                    mode='edge')
    return padded, padx, pady


def _nextpow2(i):
    buf = np.ceil(np.log(i) / np.log(2))
    return int(2 ** buf)


def _fftfreqs(x, y, shape, padshape):
    """Get two 2D-arrays with the wave numbers in the x and y directions."""
    nx, ny = shape
    dx = (x.max() - x.min()) / (nx - 1)
    # fx = numpy.fft.fftfreq(padshape[0], dx)
    fx = 2 * np.pi * np.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min()) / (ny - 1)
    # fy = numpy.fft.fftfreq(padshape[1], dy)
    fy = 2 * np.pi * np.fft.fftfreq(padshape[1], dy)
    return np.meshgrid(fy, fx)[::-1]


def geo_centers(x, y, z, labels):
    """
    Find the geometric median of each cluster.

    Args:
        x: point coordinates in the x axis
        y: point coordinates in the y axis
        z: point coordinates in th ez axis
        labels: cluster of each point, no cluster is defined as 0

    Returns:
        the center of each cluster
    """
    cluster_num = max(labels)
    centers = []
    for cInd in range(1, cluster_num + 1):
        p_x = x[labels == cInd]
        p_y = y[labels == cInd]
        p_z = z[labels == cInd]
        p_arr = np.array([p_x, p_y, p_z])
        m_arr, _ = geo_median(p_arr)
        centers.append(m_arr)
    centers = np.array(centers)
    return centers


def mean_centers(x_arr, y_arr, z_arr, labels, **kwargs):
    """
    Find the mean center of each cluster.

    Weights can be added as 1D array 'weights' in order to find a weighted mean

    Args:
        x_arr: point coordinates in x axis
        y_arr: point coordinates in y axis
        z_arr: point coordinates in z axis
        labels: cluster of each point, no cluster is defined as 0
        **kwargs: dict with 'weights' as key
        the reason this takes **kwargs is to fit the same function structure as 'geo_centers'

    Returns:
        cl_centers: centers of clusters
    """
    cluster_num = max(labels)
    cl_centers = []
    if len(kwargs) == 0:
        weights = np.ones(x_arr.shape)
    else:
        weights = kwargs['weights']
    for c_ind in range(1, cluster_num + 1):
        cl_x = x_arr[labels == c_ind]
        cl_y = y_arr[labels == c_ind]
        cl_z = z_arr[labels == c_ind]
        cl_w = weights[labels == c_ind].reshape(len(cl_x))
        w_sum = np.sum(cl_w)

        mean_x = np.sum(cl_x * cl_w) / w_sum
        mean_y = np.sum(cl_y * cl_w) / w_sum
        mean_z = np.sum(cl_z * cl_w) / w_sum
        cl_centers.append([mean_x, mean_y, mean_z])
    cl_centers = np.array(cl_centers)
    return cl_centers


def geo_iter(p_arr, m_arr, weights):
    """
    One iteration of the geo_median function.

    Args:
        p_arr: The xyz coordinates of the points
        m_arr: the geometric median
        weights: weights for each dimension

    Returns:
        new_m: updated geometric mean
        new_dist: distance after updated geometric mean between p_arr and m_arr
    """
    dist = np.linalg.norm(p_arr - m_arr) / weights
    if any(dist == 0):
        new_m = m_arr * 1.000001  # perturb slightly to check for convergence
    else:
        norm = np.sum(1 / dist)
        new_m = np.sum(p_arr / dist, 1) / norm
    new_dist = np.linalg.norm(p_arr - new_m) / weights
    return new_m, new_dist


def geo_median(p_arr, thresh=1e-4, n_test=10, weights=None):
    """
    Find the geometric median of an array of points.

    Args:
        p_arr: The xyz coordinates of the points
        thresh: threshold to stop iterations
        n_test: check if passed the threshold every n_test iterations
        weights: weights for each dimension

    Returns:
        m_arr: updated geometric mean
        dist_arr: distance array for each dimension
    """
    if weights is None:
        weights = np.ones(p_arr.shape[1])
    max_iter = 10000
    m_arr = np.mean(p_arr, 1)
    dist = np.linalg.norm(p_arr - m_arr) / weights
    dist_arr = [sum(dist * weights ** 2)]
    std = thresh
    count = -1
    while std >= thresh and count < max_iter:
        count += 1
        m_arr, dist = geo_iter(p_arr, m_arr, weights)
        if count > n_test:
            std = np.std(dist_arr[-n_test:])
        dist_arr.append(sum(dist * weights ** 2))

    return m_arr, dist_arr


def recursive_ellipse(x_in, y_in, n=None, flag=True):
    """
    Calculate the characteristic ellipse for a single cluster.

    The characteristic ellipse is found by calculating the moment of inertia of the cluster, and fitting it to the
    moment of inertia of a flat ellipse
    This process is repeated iteratively, by discarding the points outside of the ellipse until stability is achieved
    Args:
        x_in: position estimate in x axis
        y_in: position estimate in y axis
        n: number of points inside the cluster
        flag: whether iterations should be done

    Returns:
        a_rr - principal axis array
        b_arr - secondary axis array
        ang_arr - principal axis angle
    #TODO: Understand wtf is happening here.
    """
    if n is None:
        n = len(x_in)
    ixx = np.sum(y_in ** 2)
    iyy = np.sum(x_in ** 2)
    ixy = -np.sum(x_in * y_in)

    coef = 2 * ixy / (ixx - iyy)
    angle = np.arctan(coef) / 2
    "ia and ib must be positive, negative values are rounding errors"
    ia = max((2 * ixy / np.sin(2 * angle) + ixx + iyy) / 2, 0)
    b = np.sqrt(5 * ia / n)
    ib = max((-2 * ixy / np.sin(2 * angle) + ixx + iyy) / 2, 0)
    a = np.sqrt(5 * ib / n)

    x_rot = x_in * np.cos(-angle) - y_in * np.sin(-angle)
    y_rot = y_in * np.cos(-angle) + x_in * np.sin(-angle)
    isin = x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2 <= 1
    nin = sum(isin)

    if nin == 0:
        return 0, 0, 0, 0

    area = np.pi * a * b
    d = nin / area

    if nin < n and flag:
        x_in = x_in[isin]
        y_in = y_in[isin]
        a, b, d, angle = recursive_ellipse(x_in, y_in, nin, flag)

    return a, b, d, angle


def calc_residual(A, u, b):
    """Calculate the resiudal error of the equation A@u=b."""
    residual = A @ u - b
    error = np.sum(residual ** 2) / len(b)
    return error


def plot_ellipse(x, y, b, shape, est_x, est_y, labels, a_arr, b_arr, ang_arr, d_arr, pos=None, thresh=0.5,
                 show_fp=True):
    """
    Plot ellipse from points.

    Args:
        x: coordinates of measurements in the x axis
        y: coordinates of measurements in the y axis
        b: measured field in cluster points
        shape: shape of the measurements
        est_x: estimated coordinates in the x axis
        est_y: estimated coordinates in the y axis
        labels: cluster labeling for each point
        a_rr - principal axis array
        b_arr - secondary axis array
        ang_arr - principal axis angle
        d_arr: density
        pos: position of the sources (if known)
        thresh: density threshold for false positive
        show_fp: whether false positives should be shown

    """
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    fig, ax = plt.subplots()
    plt.pcolormesh(x.reshape(shape), y.reshape(shape), b.reshape(shape) - 4.5e4, cmap='PuBu')
    cb = plt.colorbar()
    cb.ax.set_title("+4.5e4 [nT]", fontsize=13)
    plt.xlabel(f"x [m]", fontsize=14)
    plt.ylabel(f"y [m]", fontsize=14)
    if pos is not None and len(pos) > 0:
        plt.plot(pos[0, :], pos[1, :], marker='x', linestyle='none', color='k')
    plt.title(f"Cut Measurement Results")
    for c_ind in range(max(labels)):
        flag = False
        if d_arr[c_ind] >= thresh:
            flag = True
        else:
            if show_fp:
                flag = True
        if flag:
            cl_x = np.mean(est_x[labels == c_ind + 1])
            cl_y = np.mean(est_y[labels == c_ind + 1])
            plt.plot(cl_x, cl_y, marker='x', color='r')
            if d_arr[c_ind] >= thresh:
                score = (1 - thresh / d_arr[c_ind]) / 2 + 0.4
                el = pat.Ellipse((cl_x, cl_y), a_arr[c_ind], b_arr[c_ind], ang_arr[c_ind] * 180 / np.pi,
                                 color=[0, score, 0.3])
            else:
                el = pat.Ellipse((cl_x, cl_y), a_arr[c_ind], b_arr[c_ind], ang_arr[c_ind] * 180 / np.pi, color='r')
            ax.add_patch(el)
    plt.show()


def plot_clusters(x, y, b, shape, est_x, est_y, labels, centers):
    """
    Plot clusters of points.

    Args:
        x: coordinates of measurements in the x axis
        y: coordinates of measurements in the y axis
        b: measured field in cluster points
        shape: shape of the measurements
        est_x: estimated coordinates in the x axis
        est_y: estimated coordinates in the y axis
        labels: cluster labeling for each point
        centers: centers of each cluster

    """
    plt.figure()
    plt.pcolormesh(x.reshape(shape), y.reshape(shape), b.reshape(shape), cmap='PuBu')
    plt.colorbar()
    plt.scatter(est_x, est_y, c=labels, s=10)
    if len(centers) > 0:
        plt.plot(centers[:, 0], centers[:, 1], 'xk')
    plt.show()
