"""Horizontal Scan class."""
import copy
from datetime import datetime
from typing import overload
import warnings

import numpy as np
import pandas as pd

from mag_utils.mag_utils.scans.scan import Scan
from mag_utils.mag_utils.visualization import plotter
from mag_utils.mag_utils.scans.base_scan import BaseScan
from mag_utils.mag_utils._consts import Sequence, docs_variables_decorator
from mag_utils.mag_utils.functional.subtract_base import subtract_base
from .interpolated_scan import InterpolatedScan
from ..interpolation.registry import interpolation_registry


class HorizontalScan(Scan):
    """object containing the data of a horizontal mag scan."""

    def __init__(self,
                 file_name: str,
                 x: Sequence,
                 y: Sequence,
                 a: Sequence,
                 b: Sequence,
                 time: [Sequence, None],
                 date: datetime.date = None,
                 interpolated_data: InterpolatedScan = None,
                 is_base_removed=False,
                 sampling_rate: float = None):
        """
        Create an instance of HorizontalScan object.

        Args:
            file_name: Path to the scan file.
            x: scan coordinate [utm].
            y: scan coordinate [utm].
            a: altitude above mean sea level [m].
            b: magnetic field [nT].
            time: time vector of datetime.time objects
            is_base_removed: set whether or not a base scan values were subtracted from the data.
            date: date.
            sampling_rate: sensor sampling rate [Hz]
            interpolated_data: interpolated data
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.a = np.asarray(a)
        self.interpolated_data = interpolated_data
        self.is_base_removed = is_base_removed

        super().__init__(file_name=file_name, b=b, time=time, date=date, sampling_rate=sampling_rate)

    # pylint: disable=C0103
    @property
    def z(self):
        return self.a

    @docs_variables_decorator(str(list(interpolation_registry.keys()))[1:-1])
    def interpolate(self, method_name: str, dist_between_points: float, inplace=False) -> 'HorizontalScan':
        """
        Interpolate the data, using different methods.

        Args:
            method_name: Which interpolation method to use from interpolation_registry, available methods: {0}.
            dist_between_points: The distance between the points in the area.
            inplace: Should the function return new magScanBase or replace the self.interpolated_data.

        Returns:
            New HorizontalScan if inplace is False, otherwise itself.
        """
        # Set up the interpolation method, calculate the mesh and interpolate the data.
        interp = interpolation_registry[method_name]()
        x_mesh, y_mesh = interp.calculate_xy_mesh(x=self.x, y=self.y, distance_between_points=dist_between_points)
        interpolated_data = interp.interpolate(self.x, self.y, self.b, x_mesh, y_mesh)

        # Check rather create a new mag scan or replace the self.interpolated_data
        if inplace:
            mag_scan = self
        else:
            mag_scan = copy.deepcopy(self)

        mag_scan.interpolated_data = interpolated_data

        return mag_scan

    def plot(self, ax=None, **kwargs):
        """
        Plot a scatter plot of the scan, With b as the color of the points.

        Examples:
            import matplotlib.pyplot as plt
            from mag_utils.mag_utils.mag_utils.mag_utils.loader import blackwidow

            # Showing the plot.
            widow_path = "some/path/scan.txt"
            scan = blackwidow.load(widow_path)
            scan.plot()

            # plotting on a custom axis
            fig, ax = plt.subplots()
            plot_data = scan.plot(ax=ax)
            fig.colorbar(plot_data)
            ax.set_title('cool beans')
            plt.show()

        Args:
            ax: matplotlib axs. If given will plot the graph on it, without showing.
                Useful for creating subplots.
            **kwargs: matplotlib scatter kwargs.

        Returns:
            None in no ax is given. The output of the scatter plot if ax is given.

        """
        return plotter.plot_horizontal_scan(self.x, self.y, self.b, ax, **kwargs)

    def plot_pso(self, target_x, target_y):

        return plotter.plot_pso_output(self.x, self.y, self.b,  float(target_x), float(target_y), ax, **kwargs)



    @overload
    def subtract_base(self, base_scan: BaseScan, inplace=False) -> 'HorizontalScan':
        ...

    @overload
    def subtract_base(self, base_b, base_t, inplace=False) -> 'HorizontalScan':
        ...

    def subtract_base(self, first_arg, second_arg=None, inplace=False) -> 'HorizontalScan':
        """
        Subtract (inplace) the base scan values from the current scan.

        Args:
            first_arg: A base_scan ot the b values of a base scan.
            second_arg: None or the timestamps values a base scan.
            inplace: Should the function return new HorizontalScan or replace the self.b.

        Returns:
            New HorizontalScan if inplace is False, otherwise itself (with subtracted b).
        """
        # handle overloading
        if isinstance(first_arg, BaseScan):
            base_b = first_arg.b
            base_t = first_arg.time
        else:
            base_b = first_arg
            base_t = second_arg

        subtracted_b = subtract_base(self.b, self.time, base_b, base_t)
        if inplace:
            mag_scan = self
        else:
            mag_scan = copy.deepcopy(self)
        mag_scan.b = subtracted_b
        mag_scan.is_base_removed = True

        return mag_scan

    def __getitem__(self, key) -> 'Scan':
        """
        Create a new sliced scan.

        Args:
            key: The key is whats inside the squared-brackets ([]).
                 When you slice (BaseScan[start:stop]) the start:stop sent as 'slice' object.
                 The key can be mask too, that means it can be list/ndarray of boolean values.

        Returns:
            New HorizontalScan - Sliced.
        """
        if self.interpolated_data is not None:
            warnings.warn("New HorizontalScan.interpolated_data set to None due slicing.")
            scan = copy.deepcopy(self)
            scan.interpolated_data = None

            return super(self.__class__, scan).__getitem__(key)

        return super().__getitem__(key)

    def append(self, other: 'HorizontalScan'):
        """
        Append HorizontalScans.

        Args:
            other: the other HorizontalScan to append to this one.

        Returns:
            A new appended HorizontalScan.
        """
        if self.is_base_removed != other.is_base_removed:
            warnings.warn("Not in both of the scans the base was subtracted.")

        return super().append(other)

    def to_dataframe(self):
        """
        Returns: a DataFrame with following columns: x, y, b, height, time.

        """
        scan_df = pd.DataFrame({"x": self.x, "y": self.y, "B": self.b, "height": self.a,
                                "time": self.time})

        if hasattr(self, "b_before_subtraction") and len(self.b_before_subtraction) > 0:  # noqa: R0902
            scan_df["original B"] = self.b_before_subtraction  # noqa: R0902

        return scan_df
