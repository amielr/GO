"""Euler Deconvolution solver, credit goes to Or Barnea."""
import numpy as np
from numpy import linalg
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from tqdm import tqdm

from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan
from mag_algorithms.base_algo import Algorithm
from .utils import derivz, derivx, derivy, plot_ellipse, plot_clusters, calc_residual, geo_centers, mean_centers, \
    recursive_ellipse


class EulerDeconvolution(Algorithm):
    """Euler Deconvolution Solver."""

    def __init__(self, num_windows=100, win_size=5, keep=0.05, si=3, eps=3, min_p=5, res=10, center='mean',
                 cluster_dim=2, plot_type=None):
        """
        Parameters for Euler Deconvolution.

        Parameters:
            num_windows: Amount of windows in each dimension, overall winNum**2 windows are used
            win_size: size of each window in meters. Windows are equally dispersed. Many big windows may overlap
            keep: fraction of solutions to save and use for clustering. Must be less than 1
            si: structural index of the source. Should be 3 for small sources and 2 for long cylinder sources
            eps: Clustering parameter, allowed distance between neighboring points [m]
            min_p: Clustering parameter, minimum neighbors for a core point
            res: minimum distance between different clusters [m]
            center: way to calculate center of cluster, 'mean' for mean of the points
                    or 'geometric' for the geometric median
            cluster_dim: 2 or 3
            plot_type: kind of plot. can be "ellipse", "clusters"
        """
        self.num_windows = num_windows
        self.win_size = win_size
        self.keep = keep
        self.si = si
        self.eps = eps
        self.min_p = min_p
        self.res = res
        self.center = center
        self.cluster_dim = cluster_dim
        self.plot_type = plot_type

        if keep > 1 or keep < 0:
            raise ValueError(f'Keep must be between 0 and 1, value given is {keep}')

    def run(self, scan: HorizontalScan, threshold=0.5):
        """
        Run the Euler Deconv algorithm on a given mag scan.

        Args:
            scan: HorizontalScan with interpolated_data (Be sure to interpolate before passing to this function)
            threshold: Threshold for ellipse plotting.

        Returns:
            {"x": numeric, "y": numeric, "d2s": numeric,
             "score": None, "name": None, "density": numeric}
             where 'density' is the density of each cluster
        """
        if scan.interpolated_data is None:
            raise ValueError('Euler Deconvolution requires that the scan has interpolated data.')

        # method assumes consistent height above ground
        height = np.mean(scan.a)
        interp_data = scan.interpolated_data
        shape = interp_data.y.shape
        x = interp_data.x.flatten()
        y = interp_data.y.flatten()
        b = interp_data.b.flatten()

        dx = derivx(x, y, b, shape)
        dy = derivy(x, y, b, shape)
        dz = derivz(x, y, b, shape)

        est_x, est_y, est_z, est_b, err_arr = self.solve_windows(x, y, np.mean(height),
                                                                 b, dx, dy, dz)
        filt_x, filt_y, filt_z, filt_b, filt_err = self.filter_est(est_x, est_y, est_z, est_b, err_arr)
        labels, centers = self.cluster(scan, filt_x, filt_y, filt_z)
        scores = self.calc_cluster_scores(filt_err, labels)

        a_arr, b_arr, d_arr, ang_arr = self.calc_ellipse(filt_x, filt_y, labels)
        if self.plot_type is not None:
            if self.plot_type == 'ellipse':
                plot_ellipse(interp_data.x, interp_data.y, interp_data.b, shape, filt_x, filt_y, labels, a_arr, b_arr,
                             ang_arr, d_arr, thresh=threshold)
            elif self.plot_type == 'clusters':
                plot_clusters(interp_data.x, interp_data.y, interp_data.b, shape, filt_x, filt_y, labels, centers)

        return {'x': [centers[i][0] for i in range(len(centers))],
                'y': [centers[i][1] for i in range(len(centers))],
                'd2s': [height - centers[i][2] for i in range(len(centers))],
                'score': scores,
                'name': None,
                'density': d_arr}

    def run_with_known_deriv(self, scan: HorizontalScan, dz, threshold=0.5):
        """
        Run the Euler Deconv algorithm on a given mag scan with a known dz.

        The given dz should basically be (B1-B2)/(H1-H2) from two seconds of close heights.

        Args:
            scan: HorizontalScan with interpolated_data (Be sure to interpolate before passing to this function)
            threshold: Threshold for ellipse plotting.
            dz: Derivative in the Z axis.

        Returns:
            centers and errors
        """
        # method assumes consistent height above ground
        height = np.mean(scan.a)
        interp_data = scan.interpolated_data
        shape = interp_data.y.shape

        x = interp_data.x.flatten()
        y = interp_data.y.flatten()
        b = interp_data.b.flatten()

        dx = derivx(x, y, b, shape)
        dy = derivy(x, y, b, shape)

        est_x, est_y, est_z, est_b, err_arr = self.solve_windows(x, y, np.mean(height),
                                                                 b, dx, dy, dz)
        filt_x, filt_y, filt_z, filt_b, filt_err = self.filter_est(est_x, est_y, est_z, est_b, err_arr)
        labels, centers = self.cluster(scan, filt_x, filt_y, filt_z)
        scores = self.calc_cluster_scores(filt_err, labels)

        a_arr, b_arr, d_arr, ang_arr = self.calc_ellipse(filt_x, filt_y, labels)

        if self.plot_type is not None:
            if self.plot_type == 'ellipse':
                plot_ellipse(interp_data.x, interp_data.y, interp_data.b, shape, filt_x, filt_y, labels, a_arr, b_arr,
                             ang_arr, d_arr, thresh=threshold)
            elif self.plot_type == 'clusters':
                plot_clusters(interp_data.x, interp_data.y, interp_data.b, shape, filt_x, filt_y, labels, centers)

        return {'x': [centers[i][0] for i in range(len(centers))], 'y': [centers[i][1] for i in range(len(centers))],
                'd2s': [height - centers[i][2] for i in range(len(centers))], 'score': scores, 'name': None,
                'density': d_arr}

    def solve_windows(self, x, y, z, b, dx, dy, dz):
        """
        Solves the equation over each window in the grid.

        Args:
            x: cartesian x coordinates of the measurements
            y: cartesian y coordinates of the measurements
            z: cartesian z coordinates of the measurements
            b: magnetic flux strength of the measurements
            dx: derivative of the magnetic flux in the x axis
            dy: derivative of the magnetic flux in the y axis
            dz: derivative of the magnetic flux in the z axis

        Returns:
            est_x: estimated source positions in the x axis
            est_y: estimated source positions in the y axis
            est_z: estimated source positions in the z axis
            est_b: estimated flux amplitudes
            err_arr = evaluated error of each guess
        """
        est_x, est_y, est_z, est_b, err_arr = [], [], [], [], []
        domain = [np.min(x), np.max(x), np.min(y), np.max(y)]
        windows = self.find_windows(domain)

        base_coeff = self.si * np.ones(len(dx))
        b_all = x * dx + y * dy + z * dz + self.si * b
        for i in tqdm(range(len(windows))):
            window = windows[i]

            "Find Solution"
            isin = (window[0] <= x) & (x <= window[1]) & (window[2] <= y) & (y <= window[3])
            A = np.array([dx[isin], dy[isin], dz[isin], base_coeff[isin]])
            A = np.transpose(A)
            b_in = b_all[isin]
            if len(b_in) < 4:
                continue
            estimates, _, _, _ = linalg.lstsq(A, b_in, rcond=None)
            error = calc_residual(A, estimates, b_in)

            est_x.append(estimates[0])
            est_y.append(estimates[1])
            est_z.append(estimates[2])
            est_b.append(estimates[3])
            err_arr.append(error)

        err_arr = np.array(err_arr)
        est_x = np.array(est_x)
        est_y = np.array(est_y)
        est_z = np.array(est_z)
        est_b = np.array(est_b)

        return est_x, est_y, est_z, est_b, err_arr

    def find_windows(self, domain):
        """
        Return the corners of the sampled windows given the corners of the measured area and the window size and number.

        The window size and number in each direction is assumed to be identical,so the overall window amount is winNum^2.
        The domain and the windows are in the format of: [min_x, max_x, min_y, max_y]

        Args:
            domain: corners of the measured area

        Returns:
            win_arr: list of window corners
        """
        x_dist = domain[1] - domain[0]
        y_dist = domain[3] - domain[2]
        center_steps = np.arange(1, self.num_windows + 1) / (self.num_windows + 1)

        # window centers in x and y directions
        x_centers = domain[0] + center_steps * x_dist
        y_centers = domain[2] + center_steps * y_dist
        y = y_centers.repeat(len(x_centers))
        x = x_centers.repeat(len(y_centers)).reshape((len(y_centers), -1)).transpose().flatten()
        centers = np.transpose([x, y])

        # corners of the windows
        min_x = centers[:, 0] - self.win_size / 2
        max_x = centers[:, 0] + self.win_size / 2
        min_y = centers[:, 1] - self.win_size / 2
        max_y = centers[:, 1] + self.win_size / 2

        win_arr = np.transpose([min_x, max_x, min_y, max_y])
        return win_arr

    def filter_est(self, est_x, est_y, est_z, est_b, err_arr):
        """
        Filter the source estimates by their errors and keep only the top percentiles.

        Args:
            est_x: estimated x coordinates
            est_y: estimated y coordinates
            est_z: estimated z coordinates
            est_b: estimated magnetic flux for each point
            err_arr: error of each estimate

        Returns:
            estimates and errors after filtering
        """
        err_sort = np.argsort(err_arr, axis=None)
        best = err_sort[int((1 - self.keep) * len(err_arr)):]

        est_x = np.array(est_x)[best]
        est_y = np.array(est_y)[best]
        est_z = np.array(est_z)[best]
        est_b = np.array(est_b)[best]
        err_arr = err_arr[best]
        return est_x, est_y, est_z, est_b, err_arr

    def cluster(self, scan, x_arr, y_arr, z_arr):
        """
        Divides the position estimates into clusters, removes false clusters and returns their centers.

        Clusters are considered insignificant if they are too small, lie outside of the domain or are too close to
        another cluster.

        Args:
            x_arr: position estimate in x axis
            y_arr: position estimate in y axis
            z_arr: position estimate in z axis

        Returns:
            labels: the cluster index of each point, 0 means no cluster
            centers: the center of each cluster, weighted according to the weights
        """
        if self.cluster_dim == 2:
            points = np.array([x_arr, y_arr]).transpose()
        elif self.cluster_dim == 3:
            points = np.array([x_arr, y_arr, z_arr]).transpose()
        else:
            raise ValueError(f'cluster_dim value: {self.cluster_dim} not in allowed values {[2, 3]}')

        cluster = DBSCAN(eps=self.eps, min_samples=self.min_p).fit(points)
        labels = cluster.labels_ + 1
        if self.center == "geometric":
            find_centers = geo_centers
        elif self.center == "mean":
            find_centers = mean_centers
        else:
            raise ValueError(f'center value: {self.cluster_dim} not in allowed values {["mean", "geometric"]}')
        "Removes false clusters"
        centers = find_centers(x_arr, y_arr, z_arr, labels)
        labels = self.join_clusters(x_arr, y_arr, z_arr, labels, find_centers)
        labels = self.cluster_trim(scan, labels, centers)

        "find centers"
        centers = find_centers(x_arr, y_arr, z_arr, labels)

        return labels, centers

    def join_clusters(self, x_arr, y_arr, z_arr, labels, find_centers):
        """
        Join clusters whose centers are too close into another one.

        Clusters are considered close if the planar distance between their centers is less than res.
        Planar distance is the distance in the xy plane.

        Args:
            x_arr: point coordinates in the x axis
            y_arr: point coordinates in the y axis
            z_arr: point coordinates in the z axis
            labels: clustering label of each point, no cluster is defined as 0
            find_centers: center finding function (geo_centers or mean_centers)

        Returns:
            labels: label for each point in the clustering, no cluster is defined as 0
        """
        cluster_num = max(labels)
        cl_centers = find_centers(x_arr, y_arr, z_arr, labels)
        ind1 = 1
        while ind1 < cluster_num:
            ind2 = ind1 + 1
            while ind2 <= cluster_num:
                dist = np.sqrt(
                    (cl_centers[ind2 - 1][0] - cl_centers[ind1 - 1][0]) ** 2 + (
                            cl_centers[ind2 - 1][1] - cl_centers[ind1 - 1][1]) ** 2)
                if dist < self.res:
                    labels[labels == ind2] = ind1
                    labels[labels > ind2] -= 1
                    cluster_num -= 1
                    cl_centers = find_centers(x_arr, y_arr, z_arr, labels)
                    ind1 -= 1
                    break
                ind2 += 1
            ind1 += 1

        return labels

    def cluster_trim(self, scan, labels, centers):
        """
        Remove insignificant and distant clusters.

        A cluster is considered insignificant if it has less
        then keep**2 of the points, and distant if it's center lies outside the domain.

        Args:
            labels: the cluster index of each point
            centers: the center of each domain

        Returns:
            labels: the adjusted cluster indices
        """
        p_arr = np.zeros(max(labels))
        cluster_num = max(labels)
        for ii in range(cluster_num):
            p_arr[ii] = sum(labels == ii + 1)

        size_order = np.argsort(p_arr)
        p_num = self.num_windows ** 2 * self.keep
        thresh = p_num * 0.05
        del_inds = []

        # convex hull
        convex_hull = Delaunay(np.array((scan.interpolated_data.x.flatten(), scan.interpolated_data.y.flatten())).T)

        for ii in range(cluster_num):
            ind = size_order[ii]
            size = p_arr[ind]
            # check if center in scan
            if convex_hull.find_simplex(centers[ind][:2]) >= 0.:
                outside_flag = False
            else:
                outside_flag = True

            if size < thresh or outside_flag:
                p_num -= size
                # thresh = p_num * self.keep
                thresh = p_num * 0.05
                labels[labels == ind + 1] = 0
                del_inds.append(ind + 1)

        for ii in range(len(del_inds)):
            ind = np.argmax(del_inds)
            del_ind = del_inds.pop(ind)
            labels[labels > del_ind] -= 1

        return labels

    @staticmethod
    def calc_ellipse(x, y, labels):
        """
        Calculate the characteristic ellipse of each cluster.

        The characteristic ellipse is found by calculating the moment of inertia of the cluster, and fitting it to the
        moment of inertia of a flat ellipse.
        If any points are outside the found ellipse, they are discarded and the calculation is redone

        Args:
            x: position estimate coordinates in x axis
            y: position estimate coordinates in y axis
            labels: cluster indices of each point

        Returns:
            a_rr: principal axis array
            b_arr - secondary axis array
            d_arr - density array
            ang_arr - angle of the main axes
        """
        c_num = np.max(labels)
        a_arr = np.ones(c_num)
        b_arr = np.ones(c_num)
        d_arr = np.ones(c_num)
        ang_arr = np.ones(c_num)
        for i in range(c_num):
            x_in = x[labels == i + 1]
            x_in = x_in - np.mean(x_in)
            y_in = y[labels == i + 1]
            y_in = y_in - np.mean(y_in)
            a, b, d, angle = recursive_ellipse(x_in, y_in)

            a_arr[i] = a
            b_arr[i] = b
            d_arr[i] = d
            ang_arr[i] = angle
        return a_arr, b_arr, d_arr, ang_arr

    @staticmethod
    def calc_cluster_scores(errors, labels):
        """
        Calculate per cluster score given elements of that cluster.

        Args:
            errors: errors of the cluster points.
            labels: cluster assignments.

        Returns:
            per cluster scores.
        """
        clusterNum = np.max(labels)
        scores = np.zeros(clusterNum)
        for ii in range(clusterNum):
            scores[ii] = np.std(errors[labels == ii + 1] / np.mean(errors))
        return scores
