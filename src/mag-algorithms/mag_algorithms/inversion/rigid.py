"""
rigid.py class.

This class contains two algorithms that perform rigid inversion on mag scans.
By rigid, we mean that both the moments direction and the dipoles positions are constant throughout the run.
"""
import matplotlib.pyplot as plt
import numpy as np

from mag_algorithms.base_algo import Algorithm
from mag_utils.algo_tools.convex_hull_grids import rectangular_convex_hull_grid
from mag_utils.algo_tools.simulations import simulate_b_ax_from_dipole

from tqdm import tqdm
from scipy.fft import rfft2, irfft2


class PointFinderInversion(Algorithm):
    """
    This class runs SGD based adaptive Lasso Least Absolute Deviation algorithm.

    Contains an adaptive L1-penalty heuristic that tries to find points
    (L0 penalty approximation heuristic) by changing the step sizes of each coefficient based on its
    influence on the current sum without changing the L1 penalty step size.
    """

    def __init__(self,
                 depth: float,
                 spacing: float = 2.,
                 l1_penalty: float = 0.01,
                 iterations: float = 5.,
                 convex_hull_factor=1.2,
                 absolute_depth: bool = False,
                 step_size=.001,
                 b_earth=None,
                 sparsify_grads_power=None):
        """
        Instantiate an PointFinderInversion object.

        Args:
            depth : float
            the depth at which the dipoles should be simulated below the scan average height (or height
            above see level if absolute_depth = True).

            spacing : float
            spacing between dipoles

            l1_penalty : float
            Lasso penalty (read about Lasso Regression).

            iterations : float
            number of iterations to pass through all data points

            convex_hull_factor : float
            by how much the dipole grid can exit the convex hull of the x,y values of the scan.

            absolute_depth : Bool
            If True depth is the altitude above see level, otherwise depth below scans mean height.

            step_size : float
            The step size of the sgd optimization.

            b_earth : np.array of length 3 or None.
            The direction of the magnetic field of Earth.

            sparsify_grads_power: float or None,
            Determines by how much the step size of the parameter that influence the most on the current sum will be
            larger than the other parameters.
        """
        self.depth = depth
        self.spacing = spacing
        self.l1_penalty = l1_penalty
        self.iterations = iterations
        self.convex_hull_factor = convex_hull_factor
        self.step_size = step_size
        self.absolute_depth = absolute_depth

        if b_earth is None:
            self.b_earth = np.array([0., 1., -1.]) / np.sqrt(2)
        else:
            self.b_earth = b_earth / np.linalg.norm(b_earth)

        if sparsify_grads_power is None:
            self.sparsify_grads_power = 10
        else:
            self.sparsify_grads_power = sparsify_grads_power

    # TODO: step size is const, change it adaptively.
    def run(self, scan, threshold=None):
        """
        Run on mag scan.

        Args:
            scan: mag_scan
            threshold: irrelevant for this algorithm.

        Returns: dict with the following data:
            {"x": np.array,
            "y": np.array,
            "z": np.array,
            "moments_amplitude": np.array,
            "dipoles_rectangle_shape": (int, int)}
        """
        b = scan.b - scan.b.mean()

        # create a rectangular grid of points, where we simulate the dipoles.
        xs, ys, n_major, n_minor = rectangular_convex_hull_grid(scan.x, scan.y, spacing=self.spacing,
                                                                convex_hull_factor=self.convex_hull_factor)

        scan_height_mean = scan.a.mean()

        # calculate absolute and relative height of the dipoles
        absolute_dipoles_depth = scan_height_mean - self.depth if not self.absolute_depth else self.depth
        dipoles_depth_below_scan = scan_height_mean - absolute_dipoles_depth

        # The moment and positions of the simulated dipoles.
        moment = self.b_earth * dipoles_depth_below_scan ** 3
        dipole_positions = np.column_stack([xs, ys, np.ones_like(xs) * absolute_dipoles_depth])

        # The scan indices order the sgd runs on.
        shuffled_indices = get_shuffled_indices(len(scan), self.iterations)

        dc = 0
        moment_amplitudes = np.zeros_like(xs)

        # start running optimizations
        for i, shuffled_i in tqdm(list(enumerate(shuffled_indices))):
            # Shuffled sample to run sgd on.
            scan_pos = np.array([scan.x[shuffled_i], scan.y[shuffled_i], scan.a[shuffled_i]])

            # The row in the Linear regression matrix that corresponds to the shuffled index of the scan. We use the
            # symmetry between the sensor and source (replacing r with -r doesn't change the formula) to compute this
            # row.
            row = simulate_b_ax_from_dipole(dipole_pos=scan_pos,
                                            dipole_moment=moment,
                                            b_ax=self.b_earth,
                                            scan_xyz=dipole_positions)

            # calculate the influence of each coeff on the current sample of the simulation
            influence = np.abs(row * moment_amplitudes) ** self.sparsify_grads_power

            # normalize such that the parameters that influences the most has step_size = step_size,
            # and the parameters that do not influence at all has step_size = 0.5 * step_size.
            influence /= max(influence.max(), 1e-15)
            step_sizes_per_dipole = (influence * self.step_size + np.ones(len(influence)) * self.step_size) / 2

            # Least Absolute Deviation error gradient of current sample (the gradient of abs is sign).
            err = np.sign(b[shuffled_i] - (row @ moment_amplitudes + dc))

            # SGD update with adaptive step sizes heuristic that tries to make the Lasso converge on points.
            moment_amplitudes += err * row * step_sizes_per_dipole

            dc += self.step_size * err

            # Perform non-negative Lasso sgd adaptation:
            moment_amplitudes -= self.l1_penalty * self.step_size
            moment_amplitudes[moment_amplitudes < self.l1_penalty * self.step_size] = 0

        return {"x": dipole_positions[:, 0],
                "y": dipole_positions[:, 1],
                "z": dipole_positions[:, 2],
                "moments_amplitude": moment_amplitudes,
                "dipoles_rectangle_shape": (n_minor, n_major)}


class LMSInversion(Algorithm):
    """
    This class runs an LMS-based algorithm with a Lasso penalty.

    Also approximately supports SVD-dimensionality reduction during the adaptation
    by using FFT and the properties of the SVD of a circular convolution matrix.
    """

    def __init__(self,
                 depth: float,
                 spacing: float = 1.,
                 iterations: float = 1.,
                 l1_penalty: float = 0.01,
                 convex_hull_factor: float = 1.5,
                 absolute_depth: bool = False,
                 init_mu: float = .1,
                 fft_cond: float = 0.,
                 b_earth=None):
        """
        Instantiate an LMSInversion object.

        Args:
            depth : float
            the depth at which the dipoles should be simulated below the scan average height
            (or height above sea level if absolute_depth = True).

            spacing : float
            spacing between dipoles

            l1_penalty : float
            Lasso penalty (read about Lasso Regression).

            iterations : float
            number of iterations to pass through all data points in the SGD optimization. Can be non integer to
            partially pass over the data points.

            convex_hull_factor : float
            by how much the dipole grid can exit the convex hull of the x,y values of the scan.

            absolute_depth : Bool
            If True depth is the altitude above see level, otherwise depth below scans mean height.

            init_mu : float
            The initial step size of the sgd optimization relative to the largest singular value.
            If a value larger than 2 is given the function might diverge.

            fft_cond : float, recommended values between 0. and 0.1.
            Uses FFT and the properties of a circular convolution matrix to approximately apply a dimensionality
            reduction on the parameters of the optimization. This is useful in the case of very small spacing to
            allow the function to converge even if the number of dipoles is orders of magnitudes more than the
            number of samples of the sensor (equations). Singular values that are smaller than fft_cond *
            largest_singular_value are removed.

            b_earth : np.array of length 3 or None.
            The direction of the magnetic field of Earth.
        """
        self.depth = depth
        self.spacing = spacing
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.convex_hull_factor = convex_hull_factor
        self.absolute_depth = absolute_depth
        self.init_mu = init_mu
        self.fft_cond = fft_cond

        if b_earth is None:
            self.b_earth = np.array([0., 1., -1.]) / np.sqrt(2)
        else:
            self.b_earth = b_earth / np.linalg.norm(b_earth)

    def run(self, scan, threshold=None):
        """
        Run on mag scan.

        Args:
            scan: mag_scan
            threshold: irrelevant for this algorithm.

        Returns: dict with the following data:
            {"x": np.array,
            "y": np.array,
            "z": np.array,
            "moments_amplitude": np.array,
            "dipoles_rectangle_shape": (int, int)}
        """
        b = scan.b - scan.b.mean()

        # create a rectangular grid of points, where we simulate the dipoles.
        xs, ys, n_major, n_minor = rectangular_convex_hull_grid(scan.x, scan.y, spacing=self.spacing,
                                                                convex_hull_factor=self.convex_hull_factor)

        scan_height_mean = scan.a.mean()

        # calculate absolute and relative height of the dipoles
        absolute_dipoles_depth = scan_height_mean - self.depth if not self.absolute_depth else self.depth
        dipoles_depth_below_scan = scan_height_mean - absolute_dipoles_depth

        scan_center = np.array([xs.mean(),
                                ys.mean(),
                                scan_height_mean])  # The point in the middle of the scan's convex hull.

        # The moment and positions of the simulated dipoles.
        moment = self.b_earth * dipoles_depth_below_scan ** 3
        dipole_positions = np.column_stack([xs, ys, np.ones_like(xs) * absolute_dipoles_depth])

        # Simulates anderson at the middle of the scan. If the scan is 2-dimensional this allows us to approximate
        # the singular values of the simulation matrix (the matrix containing all the simulation of the dipoles).
        # --------------------------------------------------------------------------------------------------------
        base_simulation = simulate_b_ax_from_dipole(scan_center,
                                                    moment,
                                                    self.b_earth,
                                                    dipole_positions).reshape(n_minor, n_major)
        singular_values = np.abs(rfft2(base_simulation))
        largest_singular_value = singular_values.max()
        # --------------------------------------------------------------------------------------------------------

        # The 2d frequencies we don't allow in the gradients of the sgd. This reduces the number of parameters.
        dim_reduction_freqs = singular_values < largest_singular_value * self.fft_cond

        init_step_size = self.init_mu / largest_singular_value  # the sgd step size.

        # TODO: smarter step size changes, see sklearn SGDRegresor (or pytorhch) for example.
        # set the step size at each part of the optimization.
        step_size_changes = {int(0.1 * len(scan.b)): init_step_size / 5,
                             int(0.5 * len(scan.b)): init_step_size / 50,
                             len(scan.b): init_step_size / 100}

        # The scan indices order the sgd runs on.
        shuffled_indices = get_shuffled_indices(len(scan), self.iterations)

        dc = 0
        moment_amplitudes = np.zeros_like(xs)
        step_size = init_step_size
        for i, shuffled_i in tqdm(list(enumerate(shuffled_indices))):
            # Choosing step size based on the step size changes dictionary.
            if i in step_size_changes:
                step_size = step_size_changes[i]

            # Shuffled sample to run sgd on.
            scan_pos = np.array([scan.x[shuffled_i], scan.y[shuffled_i], scan.a[shuffled_i]])

            # The row in the Linear regression matrix that corresponds to the shuffled index of the scan. We use the
            # symmetry between the sensor and source (replacing r with -r doesn't change the formula) to compute this
            # row.
            row = simulate_b_ax_from_dipole(dipole_pos=scan_pos,
                                            dipole_moment=moment,
                                            b_ax=self.b_earth,
                                            scan_xyz=dipole_positions)

            # Perform approximate SVD dimensionality reduction using the properties of a circular convolution matrix.
            if self.fft_cond != 0:
                row = row.reshape(n_minor, n_major)
                f = rfft2(row)
                f[dim_reduction_freqs] = 0
                row = irfft2(f, row.shape).real.ravel()

            # LMS filter (linear regression sgd) adaptation formula:
            err = b[shuffled_i] - (row @ moment_amplitudes + dc)
            moment_amplitudes += step_size * err * row
            dc += step_size * err * largest_singular_value

            # Perform non-negative Lasso sgd adaptation:
            moment_amplitudes -= self.l1_penalty * step_size
            moment_amplitudes[moment_amplitudes < self.l1_penalty * step_size] = 0

        return {"x": dipole_positions[:, 0],
                "y": dipole_positions[:, 1],
                "z": dipole_positions[:, 2],
                "moments_amplitude": moment_amplitudes,
                "dipoles_rectangle_shape": (n_minor, n_major)}


def get_shuffled_indices(num_samples: int, iterations: float):
    """
    Shuffles the indices of the input vector iterations times.

    Args:
        num_samples: number of samples in input vector.
        iterations: number of iterations the algorithm should be run over the input vector.

    Returns: an np.array of shuffled indices.
    """
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)
    shuffled_indices = np.tile(shuffled_indices, int(np.ceil(iterations)))[:int(np.ceil(iterations * num_samples))]
    np.random.shuffle(shuffled_indices)

    return shuffled_indices
