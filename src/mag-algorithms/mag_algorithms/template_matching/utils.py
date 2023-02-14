"""TemplateScan objects."""

import json
import os.path
import warnings
import datetime
import h5py
import numpy as np

import mag_utils.visualization.plotter as plotter
from mag_utils.scans.interpolated_scan import InterpolatedScan
from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.scans.labeled_scan import LabeledScan
from mag_algorithms.template_matching.consts import TOLERANCE


class TemplateScan(HorizontalScan):
    """TemplateScan."""

    def __init__(self, file_name, x, y, b, d2s, pos, axis=None, interpolated_data=None):
        """TemplateScan, sub-class of HorizontalScan."""
        zeros = np.array([datetime.time(0)] * int(x.shape[0]))
        super().__init__(file_name=file_name,
                         x=np.sort(x),
                         y=np.sort(y)[::-1],
                         a=zeros,
                         b=b,
                         time=zeros,
                         is_base_removed=False,
                         interpolated_data=interpolated_data)
        self.d2s = d2s
        self.pos = pos
        self.axis = axis

    def plot(self, ax=None, **kwargs):
        """
        Plot a scatter plot of the scan, With b as the color of the points.

        Args:
            ax: matplotlib axs. If given will plot the graph on it, without showing.
                Useful for creating subplots.
            **kwargs: matplotlib scatter kwargs.

        Returns:
            None in no ax is given. The output of the scatter plot if ax is given.

        """
        return plotter.plot_interpolated_scan(self.x, self.y, self.b, ax, **kwargs)


def interpolate_with_processing_for_tm(scan: HorizontalScan, dist_between_points=0.4):
    """
    Interpolate HorizontalScan with processing.

    Linear interpolation and fill the extrapolated values with the expanded values that
    nearest interpolation calculates.
    Than flip the y and b values to make it fit.

    Args:
        dist_between_points: The distance between points for creating the mesh for the interpolation.
        scan: The HorizontalScan.

    Returns:
        Interpolated And Processed HorizontalScan.
    """
    nearest = scan.interpolate("Nearest", dist_between_points, False)
    scan.interpolate("Linear", dist_between_points, True)
    mask = scan.interpolated_data.mask

    # Fill the null with the nearest's values
    scan.interpolated_data.b[~mask] = nearest.interpolated_data.b[~mask]

    return scan


def load_comsol_scan(scan_path):
    """
    Load comsol sim scan into mag scan's interpolated data.

    Args:
        scan_path: The path to the comsol's h5 scan.

    Returns:
        HorizontalScan.
    """
    with h5py.File(scan_path, 'r') as h5:
        data = h5.get("data")
        cols = len(np.unique(data.get("x")))
        rows = len(np.unique(data.get("y")))

        x = np.flipud(np.array(data.get("x")).reshape(cols, rows))
        y = np.flipud(np.array(data.get("y")).reshape(cols, rows))
        b = np.flipud(np.array(data.get("mfnc.normB")).reshape(cols, rows))

        interpolated_scan = InterpolatedScan(x=x,
                                             y=y,
                                             b=b,
                                             mask=np.ones([cols, rows]),
                                             interpolation_method="COMSOL")
        pos = list(json.loads(data.attrs['pos']).values())[0]

        return LabeledScan(file_name=scan_path,
                           x=x,
                           y=y,
                           d2s=np.abs(data.get("z")[0] - pos[2]),
                           pos=pos,
                           b=b,
                           interpolated_data=interpolated_scan,
                           is_real=False)


def calculate_downsample_factor(x_values, distance_between_points):
    """
    Calculate the downsample factor.

    Args:
        x_values: The unique x values.
        distance_between_points: Distance between points.

    Returns:
        Downsample factor.
    """
    # downsample if needed
    original_distance_between_points = np.round(np.abs(np.diff(x_values[:2])[0]), 1)  # meter

    # check if distance_between_points is divisible by original_distance_between_points
    if (np.abs((distance_between_points / original_distance_between_points)) % 1) > TOLERANCE:
        warnings.warn('distance between points is not a divisible by the original distance.'
                      ' downsample will not be perfect!')

    return int(distance_between_points / original_distance_between_points)


def downsample(x_values, y_values, scan, distance_between_points):
    """
    Downsample the scan.

    Args:
        x_values: The unique x values.
        y_values: The unique y values.
        scan: The unique b values.
        distance_between_points: Distance between points.

    Returns:
        Downsample'd scan if downsample factor is higher than 1, else returns the scan without changes.
    """
    downsample_factor = calculate_downsample_factor(x_values, distance_between_points)

    if downsample_factor > 1:
        scan = scan[::downsample_factor, ::downsample_factor]
        x_values = x_values[::downsample_factor]
        y_values = y_values[::downsample_factor]

    return x_values, y_values, scan


def load_template_scan(filepath, distance_between_points=0.1) -> TemplateScan:
    """
    Load template scan, process it and handle multiple heights.

    Args:
        filepath: The template's path.
        distance_between_points: Distance between points.

    Returns:
        List of processed templates for each height.
    """
    # load data from file
    with h5py.File(filepath, 'r') as h5:
        data = h5.get("data")

        # Extract data
        x_values = np.sort(np.unique(data.get("x")))
        y_values = np.sort(np.unique(data.get("y")))
        b_values = np.array(data.get("mfnc.normB"))  # T
        cols = len(x_values)
        rows = len(y_values)

        if rows * cols != len(b_values):
            raise ValueError(f'Size of b - {len(b_values)} is different than mesh size: {rows} {cols}')

        # Flip data, downsample and extract pos
        scan = np.flipud(b_values.reshape(cols, rows))
        x_values, y_values, scan = downsample(x_values, y_values, scan, distance_between_points)
        pos = list(json.loads(data.attrs['pos']).values())[0]

        return TemplateScan(file_name=os.path.basename(filepath),
                            x=x_values,
                            y=y_values,
                            b=scan,
                            d2s=np.abs(data.get("z")[0] - pos[2]),
                            pos=pos,
                            axis=None)
