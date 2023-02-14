import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from simulations.dipole_sims import create_magnetic_dipole_simulation as CLB

# set up ground one to use local module instead of pip-installed version
mag_utils_path = Path(__file__).absolute().parents[1] / "mag_utils"
mag_algorithms_path = Path(__file__).absolute().parents[1] / "mag-algorithms"

if mag_utils_path.exists() and str(mag_utils_path) not in sys.path:
    sys.path.append(str(mag_utils_path))
if mag_algorithms_path.exists() and str(mag_algorithms_path) not in sys.path:
    sys.path.append(str(mag_algorithms_path))


class MagneticObject:
    def __init__(self):
        self.moment_type = None

    def calculate_magnetic_field(self, route):
        raise NotImplementedError

    def plot_sources(self):
        # TBD: we need to implement plotter for the sources ie -- plot the tunnel you created
        raise NotImplementedError

    @staticmethod
    def convert_pos_to_cartesian(coordinates_type, vec):
        """
        :param coordinates_type: str
         "cartesian" / "spherical" / "cylindrical" / "unit"
        :param vec: np.array
        cartesian - np.array([x, y, z])
        spherical - np.array([r, phi, theta])
        cylindrical - np.array([r, theta, z])
        unit - np.array([x, y, z, magnitude])
        :return: vector in cartesian coordinates
        """
        coordinates_type = coordinates_type.lower()
        if coordinates_type == "cartesian":
            return np.array(vec)
        elif coordinates_type == "spherical":
            return np.array([vec[0] * np.sin(vec[1]) * np.cos(vec[2]),
                             vec[0] * np.sin(vec[1]) * np.sin(vec[2]),
                             vec[0] * np.cos(vec[1])
                             ])
        elif coordinates_type == "cylindrical":
            return np.array([vec[0] * np.cos(vec[1]),
                             vec[0] * np.sin(vec[1]),
                             vec[2]
                             ])
        elif coordinates_type == "unit":
            return (vec[:3] * vec[3]) / np.linalg.norm(vec[:3])
        else:
            raise TypeError(
                "Type not recognized, please check your parameters and make sure they match the documentation.")

    def get_pos(self):
        raise NotImplementedError


class MagneticDipole(MagneticObject):
    def __init__(self, p_type="cartesian", p_vec=None, m_type="cartesian", m_vec=None):
        self.pos = self.convert_pos_to_cartesian(p_type, p_vec)
        self.moment = self.convert_pos_to_cartesian(m_type, m_vec)

    def calculate_magnetic_field(self, route):
        """

        :param route: nd.array
        contains the route on which the field is calculated nd.array([[x0, y0, z0],...,[xn, yn, zn]])
        :return: nd.array
        Vector of the Magnetic Field np.array([[Bx0, By0, Bz0]...[Bxn, Byn, Bzn]])
        """
        return CLB(source_pos=np.array([self.pos] if len(self.pos.shape) < 2 else self.pos), scan_pos_mat=route,
                   source_moment=np.array([self.moment]), B_ext=0, scalar=False)

    def change_position(self, p_type="cartesian", p_vec=None):
        self.pos = self.convert_pos_to_cartesian(p_type, p_vec)

    def change_moment(self, m_type="cartesian", m_vec=None):
        self.moment = self.convert_pos_to_cartesian(m_type, m_vec)

    def get_pos(self):
        return np.array([self.pos])


class FiniteDipoleChain(MagneticObject):
    def __init__(self, p_initial=None, p_initial_type='cartesian', p_final=None, p_final_type='cartesian',
                 m=None, m_type='cartesian', m_units='Am^2', density=1.1):
        """
        :param p_initial:
        :param p_initial_type:
        :param p_final:
        :param p_final_type:
        :param m: moment per meter [Am] or moment per dipole [Am^2]
        :param m_type:
        :param density: dipoles per meter [1/m]
        """
        self.initial_position = self.convert_pos_to_cartesian(coordinates_type=p_initial_type, vec=p_initial)
        self.final_position = self.convert_pos_to_cartesian(coordinates_type=p_final_type, vec=p_final)
        self.density = density
        self.len = np.linalg.norm(self.final_position - self.initial_position)
        self.n_dipoles = round(self.len * self.density)

        self.moment = self.convert_pos_to_cartesian(coordinates_type=m_type, vec=m)
        if m_units == 'Am^2':  # if the entered moment is the moment of the whole chain
            self.dipole_moment = self.moment
        elif m_units == 'Am':  # if the entered moment is moment of dipoles/meter
            self.dipole_moment = self.moment / self.density
        self.chain = self.create_chain()

    def create_chain(self):
        pos = np.linspace(self.initial_position, self.final_position, self.n_dipoles)
        return [MagneticDipole(p_vec=loc, m_vec=self.dipole_moment) for loc in pos]

    def calculate_magnetic_field(self, route):
        return sum([dipole.calculate_magnetic_field(route) for dipole in self.chain])

    def change_density(self, density):
        self.density = density
        self.chain = self.create_chain()

    def change_moment(self, moment, m_type='cartesian', m_units='Am'):
        self.moment = self.convert_pos_to_cartesian(coordinates_type=m_type, vec=moment)
        if m_units == 'Am^2':  # if the entered moment is the moment of the whole chain
            self.dipole_moment = self.moment
        elif m_units == 'Am':  # if the entered moment is moment of dipoles/meter
            self.dipole_moment = self.dipole_moment / self.density
        for dipole in self.chain:
            dipole.change_moment(m_type=m_type, m_vec=self.dipole_moment)

    def get_pos(self):
        return np.array([dipole.pos for dipole in self.chain])

    def rotate(self, angle, axis):
        """
        :param angle: float, angle in radians
        :param axis: str, axis to rotate about (x, y or z)
        :return:
        """
        positions = self.get_pos()
        mean_vector = np.array([np.mean(positions[:, 0]), np.mean(positions[:, 1]), np.mean(positions[:, 2])])

        positions = positions - mean_vector

        if axis == "x":
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
        elif axis == "y":
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == "z":
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
        else:
            raise TypeError(f"Unrecognized axis name ({axis})")

        rotated_pos = np.matmul(R, positions.T).T + mean_vector

        for i, d in enumerate(self.chain):
            d.change_position(p_vec=rotated_pos[i])

    def plot_sources(self):
        plt.plot([self.initial_position[0], self.final_position[0]], [self.initial_position[1], self.final_position[1]])


class ConstantField(MagneticObject):
    def __init__(self, m_type='cartesian', m=np.array([0, 4.4 * 10 ** -5, -4.4 * 10 ** -5]) / np.sqrt(2)):
        self.moment = self.convert_pos_to_cartesian(m_type, m)
        self.moment_type = None

    def calculate_magnetic_field(self, route):
        return np.array([self.moment] * len(route))

    def get_pos(self):
        return np.array([None] * 3)

    def change_moment(self, m, m_type='cartesian'):
        self.moment = self.convert_pos_to_cartesian(m_type, m)


class RectangularSource(MagneticObject):
    def __init__(self, m, p_min, p_max, x_density, y_density, p_min_type='cartesian', p_max_type='cartesian',
                 m_type='cartesian'):
        """
        :param m_type: str
           "cartesian" / "spherical" / "cylindrical" / "unit"
        :param m: moment of individual dipoles
        :param p_*_type: str
           "cartesian" / "spherical" / "cylindrical" / "unit"
        :param p_min: 1x3 np.array that contains the vertex of rectangular source that has minimum x,y values
        :param p_max: 1x3 np.array that contains the vertex of rectangular source that has maximum x,y values
        :param x_density: density of dipoles in the x-direction (in dipoles/meter)
        :param y_density: density of dipoles in the y-direction (in dipoles/meter)
        """

        # convert inputs to cartesian
        self.moment_type = '[A]'
        m = self.convert_pos_to_cartesian(m_type, m)
        self.p_min = self.convert_pos_to_cartesian(p_min_type, p_min)
        self.p_max = self.convert_pos_to_cartesian(p_max_type, p_max)

        # assign more intuitive variable names
        max_x, min_x = self.p_max[0], self.p_min[0]
        max_y, min_y = self.p_max[1], self.p_min[1]
        max_z, min_z = self.p_max[2], self.p_min[2]

        # calculate amount of needed dipoles in the y direction
        self.n_dipoles_y = round((max_y - min_y) * y_density)

        # calculate y-values at which each chain will be located
        self.y_endpoints = np.linspace(min_y, max_y, self.n_dipoles_y)

        # create chains that form rectangle
        self.rect = [FiniteDipoleChain(m=m, p_initial=np.array([min_x, y_endpoint, min_z]),
                                       p_final=np.array([max_x, y_endpoint, max_z]), density=x_density) for y_endpoint
                     in self.y_endpoints]

    def calculate_magnetic_field(self, route):
        return sum([chain.calculate_magnetic_field(route) for chain in self.rect])

    def get_pos(self):
        return np.concatenate([c.get_pos() for c in self.rect])


def create_segmented_tunnel(x, y, z, m, density, dist: np.array, theta: np.array):
    """

    Parameters
    ----------
    x : x0 of the tunnel
    y : y0 of the tunnel
    z : z0 of the tunnel (constant)
    m : moment of the tunnel -- by default in Am
    density : dipoles per meter [1/m]
    dist : array (ndarray) of all the segments lengths (in accordance to theta array)
    theta : array (ndarray) of all the segments thetas -- by azimuth, 0==>north, 90==>east (in accordance to dist array)

    Returns SimulatedScan type scan of the segmented tunnel
    -------

    """
    p_initial = np.array([x, y, z])
    z_constant = p_initial[2]
    theta = 90 - theta  # from azimuth to degrees
    segments = []

    for single_dist, single_theta in zip(dist, theta):
        x_calc = np.cos(np.deg2rad(single_theta)) * single_dist
        y_calc = np.sin(np.deg2rad(single_theta)) * single_dist

        p_final = np.array([p_initial[0] + x_calc, p_initial[1] + y_calc, z_constant])

        segment = FiniteDipoleChain(p_initial=p_initial, p_final=p_final, m=m, density=density)
        segments.append(segment)

        p_initial = p_final
    return segments