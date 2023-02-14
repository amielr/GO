import numpy as np
import pandas as pd
import pyswarms as ps
from mag_utils.loader import blackwidow
from tkinter.filedialog import askopenfile
from ground_one.data_processing import magscan2df
from simulations.dipole_sims import create_scan_matrix_from_gz, create_magnetic_dipole_simulation, MU0, B_EXT

LOWER_BOUND_OFFSET = [-20, -20, -60]
UPPER_BOUND_OFFSET = [20, 20, -1]
OPTIONS = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}


class ParticleSwarmsSimulation:
    def __init__(self, n_sources=1, scan_paths=None, bounds=None, n_iterations=200, n_particles=1000,
                 options=OPTIONS, save_path=np.array([[None]]), scan_mat=None,ftol=-np.inf, ftol_iter=1):
        """
        :param n_sources: int
            number of sources to simulate and scan
        :param scan_paths: list of path, path-like or pathlib.Path strings
            the simulation will run on these scans
            if None - tk ask_files dialog
        :param bounds: tuple
            bounds for the PSO starting positions.
            bounds = (lower_bound, upper_bound)
            lower/upper bound = ndarray contains the particle's starting box's edges (x, y, z, mx, my, mz)
        :param n_iterations: int
            how many steps the particles do
        :param n_particles: int
            number of particles in the swarm
        :param options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                     cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        :param save_path: path
            where the summary of the solutions from the PSO will be saved (as xlsx)

        """

        self.save_path = save_path
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.options = options
        self.n_sources = n_sources
        self.scan_paths = scan_paths
        self.scan_mat = scan_mat
        self.ftol = ftol
        self.ftol_iter = ftol_iter

    def get_matrix(self):
        """
        for getting the scan matrix (x, y, z) and the signal (B[T]).
        :param scan_mat: if None: reading from file.
                else: scan mat = ndarray (x, y, z, B).
        :return:
        """
        if self.scan_mat is None:
            # get Gz file
            self.scan_mat = blackwidow.lab_txt_format_loader(askopenfile())
            self.scan_mat = magscan2df.convert_magscan_to_df(self.scan_mat)

        # create the scan matrix and b_scan
        self.B_scan, self.scan_mat = create_scan_matrix_from_gz(self.scan_mat)

        # normalizing magnetic field around zero so that b_ext will not affect the loss calculation
        self.B_scan = self.B_scan - self.B_scan.mean()

    def set_bounds(self, new_bounds=None, m_max=None):
        """
        :param new_bounds: tuple bounds for the PSO starting positions. bounds = ([lower_bound], [upper_bound]).
            lower/upper bound = ndarray contains the particle's starting box's edges (x, y, z, mx, my, mz).
        """
        if new_bounds is None:
            if not self.bounds:
                xyz_lower_bound = self.scan_mat.min(axis=0) + LOWER_BOUND_OFFSET
                xyz_upper_bound = self.scan_mat.max(axis=0) + UPPER_BOUND_OFFSET

                if m_max == "None" or m_max is None:
                    # m = B * r^3 / mu0 * 4pi
                    delta_B = self.B_scan.max() - self.B_scan.min()
                    delta_r = self.scan_mat[self.B_scan.argmax()] - self.scan_mat[self.B_scan.argmin()]
                    scalar_r = np.linalg.norm(delta_r)
                    m_max = (4 * np.pi) / MU0 * delta_B * scalar_r ** 3

                print('m_max: ' + str(m_max))

                # create borders by lower and upper border
                lb = np.array([*xyz_lower_bound, -m_max, 0, -m_max])
                ub = np.array([*xyz_upper_bound, m_max, m_max, 0])

                if self.n_sources != 1:
                    lb = np.array(list(lb) * self.n_sources)
                    ub = np.array(list(ub) * self.n_sources)

                self.bounds = (lb, ub)
        else:
            print("m_max: ", m_max)
            self.bounds = new_bounds

    def ps_simulate(self, n_loops):
        """
        :param n_loops: int
            number of PSO iterations
        :param n_sources:int
            number of calculated sources (for now, only one works. TBD: more than one)
        :return: best solution in every iteration, global best simulated signal
        """

        # create empty array for all solutions of all loops
        # save best [x, y, z, mx, my, mz, M, loss] for every loop where M = |mx, my, mz|
        self.solutions = np.zeros([n_loops, self.n_sources * 7 + 1])

        for i in range(n_loops):
            self.optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.n_sources * 6,
                                                     options=self.options, bounds=self.bounds, init_pos=None, ftol=self.ftol, ftol_iter=self.ftol_iter)

            cost, self.positions = self.optimizer.optimize(self.scans_loss_for_pyswarms, iters=self.n_iterations)

            self.solutions[i][:len(self.positions)] = self.positions
            # reshaping positions
            self.positions = self.positions.reshape(self.n_sources, 6)

            self.solutions[i][-self.n_sources - 1:-1] = np.linalg.norm(self.positions[:, 3:], axis=1)
            self.solutions[i, -1] = self.optimizer.swarm.best_cost

        self.solutions = pd.DataFrame(self.solutions)
        self.best_solution_idx = self.solutions[self.n_sources * 7].idxmin()

        # reshaping solutions to get x,y,z mx,my,mz values
        self.reshaped_solutions = self.solutions.T[self.best_solution_idx].values[0:-self.n_sources - 1].reshape(
            self.n_sources, 6)
        self.B_best = create_magnetic_dipole_simulation(np.array(self.reshaped_solutions[0:, 0:3]),
                                                        np.array(self.reshaped_solutions[0:, 3:]), self.scan_mat)

        self.B_best_avg = np.average(self.B_best)
        self.B_best = self.B_best - self.B_best_avg

    def scans_loss_for_pyswarms(self, particles):
        """
        Loss function that calculates the distance in Tesla between simulated scan and original scan.
        The scan (both xyz and B arrays [scan_mat and B_scan]) must be pre-declared.
        :param particles: array of n particles with 6 dimensions (x, y, z, mx, my, mz).
        loss_mat: loss for every particle.
        :return: distance (in Tesla) between the real scan and the simulated scan.
        """


        # creating a simulation
        self.loss_mat = np.zeros(particles.shape[0])
        # creating a simulation
        # reshape particles to have different cells for each source
        particles = particles.reshape(self.n_particles, int(len(particles[0]) / 6), 6)

        B_simu = 0
        for j in range(self.n_sources):
            solution_particle = particles[:, j]
            B_simu += create_magnetic_dipole_simulation(solution_particle[:, :3], solution_particle[:, 3:],
                                                        self.scan_mat, B_ext=np.array([0, 0, 0]), field_per_dipole=True,
                                                        scalar=False)

        B_simu = B_simu + B_EXT  # adding external magnetic field
        scalar_B_simu = np.linalg.norm(B_simu, axis=-1)
        scalar_B_simu = (scalar_B_simu.T - scalar_B_simu.mean(axis=-1)).T
        self.loss_mat = np.linalg.norm(self.B_scan - scalar_B_simu, axis=-1)

        return self.loss_mat


if __name__ == '__main__':
    a = ParticleSwarmsSimulation(n_sources=1, n_particles=200, n_iterations=100)
    a.get_matrix()
    a.ps_simulate(n_loops=1)
