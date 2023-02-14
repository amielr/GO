import numpy as np
from numpy.linalg import norm
from scipy.fft import rfft2, irfft2  # noqa: R0902

from mag_utils.algo_tools.simulations import simulate_b_ax_from_dipole, \
    simulate_b_ax_from_each_dipole
from mag_utils.algo_tools.np_wrappers import multiply_row_value_pairs
from mag_utils.scans.mag_scan import magScan


class SGDSandbox:  # noqa: R0902
    """
    This class is a sandbox to build SGD-based algorithms for mag.
    It should be run in a lop where each iteration, you can choose to adapt certain parameters based on scan values.
    THe class contains parameters that are used to simulate a scan and functions that can be used to optimize the
    parameters to fit the real scan.
    See Rigid Inversion for example implementation.
    """
    count_i = 0

    def __init__(self,
                 scan: magScan,
                 dipoles_positions,
                 dipoles_moments=None,
                 init_coeffs=None,
                 b_earth=None,
                 const=0):
        """
        Args:
            scan: a magScan object that contains x,y,a,b vectors
            dipoles_positions: a 3xn numpy.ndarray that contains the dipole coordinates in (x,y,z) form
            dipoles_moments: a 3xn numpy.ndarray that contains the dipole moments in (mx,my,mz) form
            init_coeffs: a vector of length n that defines the initial moment values for the optimization
            b_earth: the direction of earth's magnetic field, if different than our approximation ([0,1,-1])
            const: the constant that is added to each sample of the simulation
        """
        # set the earth field direction to approximately the right value and normalize it.
        if b_earth is None:
            b_earth = np.array([0., 1., -1.])
            b_earth = b_earth / norm(b_earth)

        # if no moments are passed, set all to be in the direction of earth's field.
        if dipoles_moments is None:
            dipoles_moments = multiply_row_value_pairs(np.tile(b_earth, (len(dipoles_positions), 1)),
                                                       np.abs(dipoles_positions.T[2] - scan.a.mean()) ** 3)

        # if no initial coeffs are given, set to zeroes.
        if init_coeffs is None:
            init_coeffs = np.zeros(len(dipoles_moments))

        # set class parameters
        # pylint: disable-next=too-many-instance-attributes
        self.dipoles_positions = dipoles_positions
        self.dipoles_moments = dipoles_moments
        self.coeffs = init_coeffs
        self.scan_gps = np.stack((scan.x, scan.y, scan.a), axis=1)
        self.scan_b = scan.b - scan.b.mean()
        self.b_earth = b_earth
        self.const = const
        self.current_coeffs = self.all_gradients = self.error = self.gradients = self.update_vec = None

    def _extract_xyz_from_pos(self, pos):
        """
        Args:
            pos: a dipole index, a list of indices or an ndarray of positions

        Returns: a matrix of gps position values

        """
        if pos is None:
            return self.scan_gps
        if isinstance(pos, (int, np.int32)):
            return np.array([self.scan_gps[pos]])
        if isinstance(pos, list) or pos.ndim == 1:
            return self.scan_gps[pos]
        if pos.ndim == 2:
            return pos
        raise Exception("Position is not good at all you bitch!")

    def get_simulation_matrix(self, pos, sensor_axis=None):
        """
        Args:
            pos: The position/s of the sensor
            sensor_axis: the axis along which the field is calculated (or projected on).

        Returns: a matrix H, such that H @ self.coeffs is the simulation at pos.

        """
        if sensor_axis is None:
            sensor_axis = self.b_earth
        points = self._extract_xyz_from_pos(pos)
        simulations = np.array(
            [simulate_b_ax_from_dipole(dipole_pos, dipole_moment, sensor_axis, points) for dipole_pos, dipole_moment
             in zip(self.dipoles_positions, self.dipoles_moments)]).T
        return simulations

    def get_simulation_row(self, pos, sensor_axis=None):
        """
        Args:
            pos: The position/s of the sensor
            sensor_axis: the axis along which the field is calculated (or projected on).

        Returns: an np.array h, such that h @ self.coeffs is the simulation at pos.

        """
        if sensor_axis is None:
            sensor_axis = self.b_earth
        point = self.scan_gps[pos]
        return simulate_b_ax_from_each_dipole(self.dipoles_positions, self.dipoles_moments, sensor_axis, point)

    def simulate_at(self, pos, sensor_axis=None):
        """
        Args:
            pos: The position/s of the sensor
            sensor_axis: the axis along which the field is calculated (or projected on).

        Returns: the simulation at pos.

        """
        return self.get_simulation_matrix(pos, sensor_axis) @ self.coeffs

    def coeffs_adaptation(self,
                          data_index,
                          step_size,
                          adapt_indices=slice(None, None),
                          update=True,
                          power=1,
                          sensor_axis=None,
                          dc_step_size=None,
                          fft_cond=0,
                          fft_shape=None,
                          step_size_func=None,
                          **step_size_kwargs):
        """
        Args:
            fft_cond: zeroes out freqs whose amplitude is smaller than the freq with the highest amplitude.
            fft_shape: the shape of the 2d fft matrix
            data_index: the sample index from scan to analyze.
            step_size: a float or np.array that represents the step size of each parameter in the sgd.
            adapt_indices: the dipole indices to adapt in this iteration.
            update: bool, if the values should be updated in place or not (the update vector is returned anyways)
            power: the power by which the error is raised in the loss function
            sensor_axis: the axis along which the field is calculated (or projected on).
            dc_step_size: the value by which the const will change each iteration.
            step_size_func: if the step size depends on any of the calculation parameters,
            a function that recieves (self, step_size, gradients, adapt_indices, **step_size_kwargs) and
            returns the step size can be used.
            Pass in a function handle if needed.
            **step_size_kwargs: other parameters that might be needed for step size func.

        Returns: vector added to coeffs, gradients, error value (simulation - real), step size

        """
        # pylint: disable=too-many-locals

        # get relevant coeffs
        self.current_coeffs = self.coeffs[adapt_indices]

        # calculate simulation to be used as gradients
        self.all_gradients = self.get_simulation_row(data_index, sensor_axis)
        if fft_shape is not None:
            self.all_gradients = self.all_gradients.reshape(fft_shape)
            for i, grad in enumerate(self.all_gradients):
                freqs = np.array(rfft2(grad))
                f_amp = np.abs(freqs)
                freqs[f_amp < f_amp.max() * fft_cond] = 0
                if power == 2:
                    # TODO: allow adaptive step size
                    step_size = .001 / f_amp.max()
                self.all_gradients[i] = irfft2(freqs, grad.shape)
        self.all_gradients = self.all_gradients.ravel()

        # calculate error
        self.error = (self.scan_b[data_index] - self.all_gradients @ self.coeffs - self.const)

        # raise error to power - 1 and keep the sign as it is a derivative
        self.error = np.sign(self.error) * abs(self.error) ** (power - 1)

        # multiply all gradients by error to get the opposite of the gradient of the loss function.
        self.gradients = self.all_gradients[adapt_indices] * self.error

        if dc_step_size is None:
            dc_step_size = np.array(step_size).mean()
        # figure out step size for each parameter
        if isinstance(step_size, float):
            step_size = np.ones(len(self.current_coeffs)) * step_size
        if step_size_func is not None:
            step_size = step_size_func(self, step_size, self.gradients, adapt_indices, **step_size_kwargs)

        # update const according to error and const step size
        self.const += self.error * dc_step_size

        # multiply gradients by step size to get the update vector
        self.update_vec = self.gradients * step_size

        # update coeffs according to update vector
        if update:
            self.coeffs[adapt_indices] += self.update_vec

        return self.update_vec, self.all_gradients, self.error, step_size

    def lasso_adaptation(self, step_size, l1_penalty, adapt_indices=slice(None, None), cutoff=1.):
        """
        Args:
            step_size: a float or np.array that represents the step size of each parameter in the sgd.
            l1_penalty: Lasso penalty (read about lasso regression).
            adapt_indices: the dipole indices to adapt in this iteration.
            cutoff: the threhold at which the coeff will be zeroed out.

        Returns: None

        """
        # calculate the sgd step of the lasso penalty
        penalty = step_size * l1_penalty
        self.coeffs[adapt_indices] -= penalty * np.sign(self.coeffs[adapt_indices])

        # zero out values that are too close to 0 at which the gradient is undefined.
        self.coeffs[adapt_indices][np.abs(self.coeffs[adapt_indices]) < cutoff * penalty] = 0

    def leakage_adaptation(self, step_size, l2s_penalty, adapt_indices=slice(None, None)):
        """
        Args:
            step_size: a float or np.array that represents the step size of each parameter in the sgd.
            l2s_penalty: Ridge penalty (read about ridge regression or Tinkhonov regularization or leaky lms).
            adapt_indices: the dipole indices to adapt in this iteration.

        Returns:

        """
        # calculate the sgd step of the ridge penalty
        penalty = step_size * l2s_penalty

        # adapt coefficients according to penalty
        self.coeffs[adapt_indices] *= (1 - penalty)
