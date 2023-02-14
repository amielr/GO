"""Simulate dipole."""
import numpy as np

from mag_algorithms.pso.consts import MU0, B_EXT


class Simulation:
    """Simulation."""

    def __init__(self, scan_route, field_positions, field_power):
        """
        Simulate mag.

        Args:
            scan_route: route of scan
            field_positions: array of [[x1,y1,z1],[x2,y2,z2], ....] of source
            field_power: array of [[mx, my, mz]] of all sources or [[mx0, my0, mz0], [mx1, my1, mz1], ... , [mxn, myn, mzn]] for each source
        """
        self.scan_route = scan_route
        self.field_positions = field_positions
        self.field_power = field_power

        if self.field_positions.shape != self.field_power.shape:
            raise ValueError("The field_power amd field_positions isn't fitted ")

    def magnetic_dipole_formula(self):
        """Magnetic equation(who the fuck use einsum)."""
        r = self.scan_route[np.newaxis, :, :] - self.field_positions[:, np.newaxis, :]
        r_norms = np.linalg.norm(r, axis=-1)

        r_normalized = r * (1 / r_norms)[:, :, np.newaxis]

        dist_moment_dot_prod = np.einsum('ijk,ik->ij', r_normalized, self.field_power)  # Should be same as:
        # dist_moment_dot_prod = np.sum(r_normalized * self.field_power[:, np.newaxis, :], axis=2)

        first_part = 3 * dist_moment_dot_prod[:, :, np.newaxis] * r_normalized
        magnetic_field_vector = (MU0 / (4 * np.pi)) * \
                                (first_part.transpose(1, 0, 2) - self.field_power).transpose(2, 1, 0) / (r_norms ** 3)
        return magnetic_field_vector.transpose(1, 2, 0)  # particle, sample, Bxyz

    def generate_simulation(self, field_per_dipole=False, scalar=True):
        """
        Generate simulation.

        Args:
            scalar: get the magnetic field in scalar
            field_per_dipole: get the field seprate by dimation

        return scan like array of B simulation (magnetic field) given parameters
        """
        # calculate B for each source
        B = self.magnetic_dipole_formula()

        if scalar:
            if field_per_dipole:
                magnetic_field = B + B_EXT
                return np.linalg.norm(magnetic_field, axis=2)
            else:
                B = B.sum(axis=0)  # B = sum of magnetic fields created by all dipoles
                magnetic_field = B + B_EXT
                return np.linalg.norm(magnetic_field, axis=1)
        else:
            if field_per_dipole:
                magnetic_field = B + B_EXT
                return magnetic_field
            else:
                B = B.sum(axis=0)  # B = sum of magnetic fields created by all dipoles
                magnetic_field = B + B_EXT
                return magnetic_field
