"""PSO credit to ofri mike from gaza."""
import numpy as np
import pyswarms as ps

from mag_algorithms.mag_algorithms.base_algo import Algorithm
from mag_algorithms.mag_algorithms.loss import Loss, L2Loss
from mag_algorithms.mag_algorithms.pso.consts import OPTIONS, LOWER_BOUND_OFFSET, UPPER_BOUND_OFFSET, MU0, B_EXT
from mag_algorithms.mag_algorithms.pso.simulate_dipole import Simulation
from mag_algorithms.mag_algorithms.pso.utils import normalize_mag_scan
from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan


class ParticleSwarmOptimization(Algorithm):
    """Particle Swarms Optimization."""

    def __init__(self, n_sources: int = 1, bounds: tuple = None,
                 pso_iterations: int = 1,
                 n_iterations: int = 200, n_particles: int = 100, loss: Loss = L2Loss(reduction='none'),
                 verbose: bool = False,
                 ftol=1e-10,
                 ftol_iter=50,
                 options=OPTIONS):
        """
        PSO.

        Args:
            n_sources: number of sources to simulate and scan
            bounds: bounds for the PSO starting positions.
            bounds = (lower_bound, upper_bound)
            lower/upper bound = contains the particle's starting box's edges (x, y, z, mx, my, mz).
            pso_iterations: if u want to run pso multiple time and get the best score.
            n_iterations: how many steps the particles do
            n_particles: number of particles in the swarm
            loss: Loss function to use.
            verbose: if True, show progress bar and output log.
            options: pso parameters (if you want to expend yore knowledge go to consts.py
                         and read the docs of OPTIONS).
            ftol: if pso run for ftol_iters number of times and gets loss below ftol the pso will stop
            ftol_iter: the number of iterations in pso runs before stopping under ftol
        """
        self.pso_iterations = pso_iterations
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.options = options
        self.n_sources = n_sources
        self.scan_route = None
        self.scan_b = None
        self.loss = loss
        self.verbose = verbose
        self.solutions = np.zeros([pso_iterations, n_sources * 7 + 1])
        self.ftol = ftol
        self.ftol_iter = ftol_iter

    def run(self, mag_scan: HorizontalScan, plot=True):
        """
        Run on mag scan.

        Args:
            mag_scan: mag_scan
            plot: flag piloting simulation

        Returns:
            for example in case of 1 resource [x,y,z,mx,my,mz], shape (n_resource,6).
        """
        self.scan_b, self.scan_route = normalize_mag_scan(mag_scan)
        self.set_bounds()
        index_and_measurement, b_simulation = self.optimize()

        if plot:
            temp_scan = mag_scan
            temp_scan.b = b_simulation
            temp_scan.plot()

        return {'x': index_and_measurement[0:, 0],
                'y': index_and_measurement[0:, 1],
                'd2s': np.mean(mag_scan.a) - index_and_measurement[0:, 2],  # need to fix.
                'score': None,
                'name': None,
                'measurement': index_and_measurement[0:, 3:]}

    def set_bounds(self, new_bounds: (np.ndarray, np.ndarray) = None
                   , m_max=None):
        """
        Set the bounds of pso optimizations.

        Args:
            new_bounds: new_bounds: tuple bounds for the PSO starting positions, bounds = ([lower_bound], [upper_bound])
            lower/upper bound  contains the particle's starting box's edges (x, y, z, mx, my, mz).
            m_max: max magnetic field
        """
        if new_bounds is None:
            if not self.bounds:
                xyz_lower_bound = self.scan_route.min(axis=0) + LOWER_BOUND_OFFSET
                xyz_upper_bound = self.scan_route.max(axis=0) + UPPER_BOUND_OFFSET

                if m_max == "None" or m_max is None:
                    # m = B * r^3 / mu0 * 4pi
                    delta_B = self.scan_b.max() - self.scan_b.min()
                    delta_r = self.scan_route[self.scan_b.argmax()] - self.scan_route[self.scan_b.argmin()]
                    scalar_r = np.linalg.norm(delta_r)
                    m_max = (4 * np.pi) / MU0 * delta_B * scalar_r ** 3

                # create borders by lower and upper border
                lb = np.array([*xyz_lower_bound, -m_max, -m_max, -m_max])
                ub = np.array([*xyz_upper_bound, m_max, m_max, m_max])

                lb = np.array(list(lb) * self.n_sources)
                ub = np.array(list(ub) * self.n_sources)

                self.bounds = (lb, ub)
        else:
            self.bounds = new_bounds

    def optimize(self):
        """
        Optimize the loss function.

        Returns:
            best optimal_solution in every iteration, global best simulated signal
        """
        # re-initialize the solutions so it wont break
        self.solutions = np.zeros([self.pso_iterations, self.n_sources * 7 + 1])
        for i in range(self.pso_iterations):
            optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.n_sources * 6,
                                                options=self.options, bounds=self.bounds, init_pos=None, ftol=self.ftol,
                                                ftol_iter=self.ftol_iter)

            cost, optimal_solution = optimizer.optimize(self.calculate_loss, iters=self.n_iterations,
                                                        verbose=self.verbose)

            self.solutions[i][:len(optimal_solution)] = optimal_solution
            # reshaping positions
            optimal_solution = optimal_solution.reshape(self.n_sources, 6)
            # insert to solutions new column of total m (linalg of mx,my,mz)
            M = np.linalg.norm(optimal_solution[:, 3:], axis=1)
            self.solutions[i, - self.n_sources - 1:-1] = M
            self.solutions[i, -1] = optimizer.swarm.best_cost

        best_solution_idx = self.solutions[:, -1].argmin()
        # reshaping solutions to get x,y,z mx,my,mz values
        index_and_measurement = self.solutions[best_solution_idx, 0:-self.n_sources - 1].reshape(
            self.n_sources, 6)

        simulation_of_optimal_value = Simulation(self.scan_route, index_and_measurement[0:, 0:3],
                                                 index_and_measurement[0:, 3:])
        simulation_b = simulation_of_optimal_value.generate_simulation()

        return index_and_measurement, simulation_b

    def calculate_loss(self, particles):
        """
        Loss function.

        Calculates the distance in Tesla between simulated scan and original scan
        the scan (both xyz and B arrays [scan_mat and B_scan]) must be pre-declared.

        Args:
            particles: array of n particles with 6 dimensions (x, y, z, mx, my, mz),loss_mat: loss for every particle.

        Returns:
            distance (in Tesla) between the real scan and the simulated scan.
        """
        # reshape all the source in one dimantion to make the simulation simple
        particles = particles.reshape(-1, 6)
        simulation = Simulation(self.scan_route, particles[:, :3], particles[:, 3:])
        b_simulation_vector = simulation.magnetic_dipole_formula()
        b_simulation_vector = b_simulation_vector.reshape(self.n_particles, self.n_sources, self.scan_route.shape[0],
                                                          -1)
        b_simulation_vector = b_simulation_vector.sum(axis=1) + B_EXT  # sum all the source
        b_simulation_scalar = np.linalg.norm(b_simulation_vector, axis=-1)
        b_simulation_scalar = (b_simulation_scalar.T - b_simulation_scalar.mean(axis=-1)).T
        loss = self.loss.forward(self.scan_b, b_simulation_scalar)
        return loss
