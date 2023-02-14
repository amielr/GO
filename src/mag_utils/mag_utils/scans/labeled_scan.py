"""The HorizontalScan with additional information (Labeled)."""
# pylint: disable=too-many-arguments
from dataclasses import dataclass
from datetime import datetime
import warnings
from typing import List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from mag_utils.mag_utils._consts import Sequence


@dataclass
class Target:
    """Contains information about a target relative to the sensor."""

    type: str  # consider an enum like object here. 17.8.22: no need for now.
    pos: Sequence
    axis: Sequence

    def __post_init__(self):
        """Input checks."""
        if len(self.pos) != 3:
            raise ValueError(f'pos should have 3 elements. got {len(self.pos)} instead.')

    def to_h5_group(self, group: h5py.Group):
        """
        Insert the class into a given h5 group.

        Args:
            group: an empty h5 group
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (list, tuple, np.ndarray)):
                group.create_dataset(attr_name, data=attr_value)
            else:
                group.attrs.create(attr_name, attr_value)

    @classmethod
    def from_h5_group(cls, group):
        """
        Create Target instance from h5 group.

        Args:
            group: target h5 group.
        """
        return cls(group.attrs['type'],
                   group['pos'][:].tolist(),
                   group['axis'][:].tolist())

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.type == other.type and np.allclose(self.pos, other.pos) and np.allclose(self.axis, other.axis)

        raise TypeError(f"Can't compare object of type {other.__class__} to {self.__class__}")


class LabeledScan(HorizontalScan):
    """
    Object containing the data of labeled HorizontalScan.

    Inherits from HorizontalScan with additional attributes for known information.
    """

    def __init__(self,
                 file_name: str,
                 x: Sequence,
                 y: Sequence,
                 a: Sequence,
                 b: Sequence,
                 time: Sequence,
                 is_base_removed: bool,
                 targets: List[Target],
                 is_real: bool,
                 sensor_type: 'str',
                 date: datetime.date = None,
                 interpolated_data: InterpolatedScan = None):
        """
        Create an instance of LabeledScan object.

        Args:
            From HorizontalScan:
                file_name: Path to the scan file.
                x: scan coordinate [utm].
                y: scan coordinate [utm].
                a: altitude above mean sea level [m].
                b: magnetic field [nT].
                time: time vector of datetime.time objects
                date: date.
                interpolated_data: interpolated data

            targets: list of targets in the scan.
            is_real: This labeled scan is real or sim.
        """
        super().__init__(file_name=file_name,
                         x=x,
                         y=y,
                         a=a,
                         b=b,
                         time=time,
                         is_base_removed=is_base_removed,
                         date=date,
                         interpolated_data=interpolated_data)

        # Additional source information
        self.targets = targets
        self.is_real = is_real
        self.sensor_type = sensor_type

    # pylint: disable=W0221
    def plot(self, ax=None, plot_labels=True, **kwargs):
        """
        Plot the targets of labeled data on scan.

        Args:
            ax: ax
            plot_labels: flag if to plot
            **kwargs: kwargs
        """
        if ax is None:
            _, ax = plt.subplots()

        super().plot(ax=ax, **kwargs)

        if plot_labels:
            target_x = [target.pos[0] for target in self.targets]
            target_y = [target.pos[1] for target in self.targets]
            ax.scatter(target_x, target_y, c='black')

        plt.show()

    @classmethod
    def label(cls,
              scan: HorizontalScan,
              labels: List[Target],
              sensor_type: str,
              is_real: bool = True) -> 'LabeledScan':
        """
        Add labels to a scan.

        Args:
            scan: HorizontalScan to add labels to
            labels: a list of Target objects.
            is_real: whether the scan is real.

        Returns:
            labeled_scan: LabeledScan object.
        """
        scan = cls(scan.file_name, scan.x, scan.y, scan.a, scan.b, scan.time,
                   scan.is_base_removed,
                   labels, is_real, sensor_type, scan.date, scan.interpolated_data)
        return scan

    def __getitem__(self, key) -> 'LabeledScan':
        """
        Create a new sliced labeled scan.

        Args:
            key: The key is whats inside the squared-brackets ([]).
                 When you slice (LabeledScan[start:stop]) the start:stop sent as 'slice' object.
                 The key can be mask too, that means it can be list/ndarray of boolean values.

        Returns:
            New LabeledScan - Sliced.
        """
        if self.interpolated_data is not None:
            warnings.warn("New LabeledScan.interpolated_data set to None due slicing.")

        scan = LabeledScan(file_name=self.file_name,
                           x=self.x[key],
                           y=self.y[key],
                           a=self.a[key],
                           b=self.b[key],
                           time=self.time[key],
                           is_base_removed=self.is_base_removed,
                           is_real=self.is_real,
                           targets=self.targets,
                           sensor_type=self.sensor_type,
                           date=self.date)

        if len(scan) == 1:
            scan.sampling_rate = None

        return scan
