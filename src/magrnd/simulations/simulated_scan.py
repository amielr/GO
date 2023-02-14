import warnings
import tkinter.filedialog
import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from simulations.magnetic_objects import MagneticDipole
from datetime import datetime, timedelta
from mag_utils.scans import magScan, HorizontalScan
from mag_utils.interpolation.registry import interpolation_registry
from simulations.magnetic_objects import ConstantField
from copy import deepcopy

DISPLAY_OFFSET = np.array([[-5, -5, -2],
                           [5, 5, 2]])
T2nT = 10 ** 9
B_EARTH_NORMALIZED = np.array([0, 1, -1]) / np.sqrt(2)


class SimulatedScan:
    def __init__(self):
        self.scan_route = []
        self.scan_sources = []
        self.b = None
        self.prev_route_state = None
        self.bg_scan_values = None

    def add_source(self, magnetic_object):
        """
        add a source to the scan list of sources
        :param magnetic_object: a MagneticObject instance
        """
        if isinstance(magnetic_object, list):
            self.scan_sources.extend(magnetic_object)
        else:
            self.scan_sources.append(magnetic_object)

    def check_route_validity(self):
        for i, source in enumerate(self.scan_sources):
            source_position = source.get_pos()
            if isinstance(source, ConstantField):
                continue
            for route in self.scan_route:
                if route[:, 0].min() > source_position[:, 0].min() or route[:, 0].max() < source_position[:, 0].max():
                    warnings.warn(f"The source ({i}) is outside the route in the X-axis.")

                if route[:, 1].min() > source_position[:, 1].min() or route[:, 1].max() < source_position[:, 1].max():
                    warnings.warn(f"The source ({i}) is outside the route in the Y-axis.")

    def remove_source(self, index):
        """
        remove a source from the scan list of sources
        :param index: index of source to remove from the scan
        """
        self.scan_sources.pop(index)

    def get_sources_pos(self):
        return np.array([source.get_pos() for source in self.scan_sources])

    def add_route(self, route_mat):
        self.scan_route.append(route_mat)

    def clear_route(self):
        # remove all route elements
        self.scan_route = []

    def update_route(self, route_mat: np.ndarray):
        """
        Args:
            route_mat: a 3xn matrix that contains each point in the route


        """
        self.clear_route()
        self.add_route(route_mat)

    @property
    def joint_route(self):
        """
        get the accumulated route of the scan
        :return: NX3 numpy array that contains the full route
        """
        return np.concatenate(self.scan_route)

    def calculate_magnetic_field(self, joint_route=True):
        """
        calculates the magnetic field in the route
        :param joint_route: Bool
        if true, the function returns list of magnetic fields per route
        else, returns np.array of the magnetic field of the whole scan route
        :return: B_scalar (with dependence on route_wise)
        """
        if joint_route:
            routes = [self.joint_route]
        else:
            routes = self.scan_route

        b_scalar_aggregation = []
        for route in routes:
            B_vec_of_scans = sum([object.calculate_magnetic_field(route) for object in self.scan_sources])

            # project the simulation vector on Earth's field
            B_scalar = B_vec_of_scans @ B_EARTH_NORMALIZED * T2nT
            b_scalar_aggregation.append(B_scalar)

        simulated_b = b_scalar_aggregation[0] if len(b_scalar_aggregation) == 1 else b_scalar_aggregation

        if len(routes) == 1 and not joint_route:
            simulated_b = np.array([simulated_b])

        if self.bg_scan_values is not None:
            simulated_b += self.bg_scan_values

        return simulated_b

    def plot(self, display_scan=True,
             display_sources=True,
             display_magnetic_field=True,
             ax=None):
        """
        :param display_scan: boolean, display the scan route
        :param display_sources: boolean, display the magnetic sources
        :param display_magnetic_field: boolean, display the magnetic field at the routes
        :param ax: if provided, plot on given ax, else create one
        """
        self.check_route_validity()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        route = self.joint_route

        mins, maxs = np.min(route, axis=0), np.max(route, axis=0)

        if display_sources:
            for source in self.scan_sources:
                # do not display constant field
                if isinstance(source, ConstantField):
                    continue

                pos = source.get_pos()

                source_mins, source_maxs = np.min(pos, axis=0), np.max(pos, axis=0)
                mins, maxs = np.minimum(source_mins, mins), np.maximum(source_maxs, maxs)

                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

        if display_magnetic_field:
            B_whole = self.calculate_magnetic_field()
            v_min, v_max = min(B_whole), max(B_whole)

            B_route_wise = self.calculate_magnetic_field(joint_route=False)
            max_value, max_cmap = None, None

            for i, route in enumerate(self.scan_route):
                field_per_route = B_route_wise[i]
                plot = ax.tricontourf(route[:, 0], route[:, 1], field_per_route, zdir='z',
                                      offset=np.mean(route[:, 2]) - 1,
                                      vmin=v_min, vmax=v_max, cmap='jet', levels=30)

                # check if max_value is defined, if not, define it, else if current value is greater than current max,
                # replace the current max with current value, save also cmap with each comparison
                if max_value is None:
                    max_value = plot.get_array().max()
                    max_cmap = plot
                elif plot.get_array().max() > max_value:
                    max_value = plot.get_array().max()
                    max_cmap = plot

            ax.figure.colorbar(max_cmap, extendfrac=[v_min, v_max], label='Magnetic Field [nT]')

        if display_scan:
            for route in self.scan_route:
                ax.plot(route[:, 0], route[:, 1], route[:, 2])

        # set display limits
        limits = np.vstack([mins, maxs]) + DISPLAY_OFFSET
        ax.set_xlim(*limits[:, 0])
        ax.set_ylim(*limits[:, 1])
        ax.set_zlim(*limits[:, 2])

        # set labels
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_zlabel('Up [m]')

        plt.show()

    def save(self, save_path=None):
        """
        Save the simulation in a format GroundZero can read
        :param save_path: where to save output csv file
        """
        if not save_path:
            fh = tk.Tk()
            fh.withdraw()
            save_path = tk.filedialog.asksaveasfilename(title="Select file")
            fh.destroy()

        B = self.calculate_magnetic_field()

        route = self.joint_route
        df = pd.DataFrame(np.array([route[:, 0], route[:, 1], B, route[:, 2],
                                    np.linspace(0, 0.1 * len(B), len(B))]).T)
        df = df.rename(columns={0: 'x', 1: 'y', 2: 'B', 3: 'height', 4: 'time'})
        df.to_csv(save_path, sep='\t', index=False)
        print("Saved!")

    def to_mag_scan(self):
        route = self.joint_route

        start_time = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
        time_vec = np.array([(start_time + timedelta(seconds=0.1 * i)).time() for i in range(route.shape[0])])
        return HorizontalScan(file_name="simulated_scan",
                              x=route[:, 0],
                              y=route[:, 1],
                              a=route[:, 2],
                              b=self.calculate_magnetic_field(),
                              time=time_vec,
                              is_base_removed=False)

    def add_background_scan(self, scan: HorizontalScan, route_matrix=None, interp_method_name="Linear"):
        centered_scan = deepcopy(scan)
        centered_scan.x -= centered_scan.x.mean()
        centered_scan.y -= centered_scan.y.mean()
        centered_scan.a -= centered_scan.a.mean()

        if route_matrix is None:
            route_matrix = np.column_stack([centered_scan.x, centered_scan.y, centered_scan.a])
            self.bg_scan_values = centered_scan.b
        else:
            interp = interpolation_registry[interp_method_name]()
            self.bg_scan_values = interp.interpolate(centered_scan.x, centered_scan.y, centered_scan.b,
                                                     x_mesh=route_matrix[:, 0],
                                                     y_mesh=route_matrix[:, 1]).b
        self.clear_route()
        self.add_route(route_matrix)


if __name__ == "__main__":
    from mag_utils.loader.main_load import load
    from routes import add_spiral_route
    from routes import add_rectangular_route

    scan = SimulatedScan()

    route1 = add_rectangular_route(x0=-30, x1=30, y0=-30, y1=30, z=7, v=5, fs=10)
    scan.add_route(route1)

    d1 = MagneticDipole(p_vec=[0, 0, -5], m_vec=[0, 7.07197591, -1.29192397])

    gz_scan = load("<insert path here>")
    scan.add_background_scan(gz_scan)

    scan.add_source(d1)

    scan.calculate_magnetic_field()
    scan.plot()
