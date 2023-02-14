"""AluminumMan."""
import warnings
from datetime import datetime
import numpy as np

from mag_utils.mag_utils.functional.calibration import calibration
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.mag_utils._consts import Sequence


class AluminumManScan(HorizontalScan):
    """
    Object containing the data of aluminum man.

    Inherits from magScan with additional attributes for known information.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 file_name: str,
                 x: Sequence,
                 y: Sequence,
                 a: Sequence,
                 bx: Sequence,
                 by: Sequence,
                 bz: Sequence,
                 time: Sequence,
                 is_base_removed: bool,
                 date: datetime.date = None,
                 interpolated_data: InterpolatedScan = None,
                 is_calibrated=False
                 ):
        """
        Create an instance of AluminumMan object.

        Args:
            From magScan:
                file_name: Path to the scan file.
                x: scan coordinate [utm].
                y: scan coordinate [utm].
                a: altitude above mean sea level [m].
                time: time vector of datetime.time objects
                date: date.

            bx: magnetic field in axis x.
            by: magnetic field in axis y.
            bz: magnetic field in axis z.
        """
        b = None

        if not is_calibrated:
            b = calibration(bx, by, bz)

        super().__init__(file_name=file_name,
                         x=x,
                         y=y,
                         a=a,
                         b=b,
                         time=time,
                         is_base_removed=is_base_removed,
                         date=date,
                         interpolated_data=interpolated_data)

        self.bx = bx
        self.by = by
        self.bz = bz

    def __eq__(self, other):
        """
        Compare between 2 AluminumMan.

        Args:
            other: the other aluminum scan to compare with.
        """
        is_equal = super().__eq__(other)

        return all([is_equal,
                    np.allclose(self.bx, other.bx),
                    np.allclose(self.by, other.by),
                    np.allclose(self.bz, other.bz)])

    def __getitem__(self, key) -> 'AluminumManScan':
        """
        Create a new sliced aluminum man.

        Args:
            key: The key is whats inside the squared-brackets ([]).
                 When you slice (AluminumMan[start:stop]) the start:stop sent as 'slice' object.
                 The key can be mask too, that means it can be list/ndarray of boolean values.

        Returns:
            New AluminumMan - Sliced.
        """
        if self.interpolated_data is not None:
            warnings.warn("New magScan.interpolated_data set to None due slicing.")

        scan = AluminumManScan(file_name=self.file_name,
                               x=self.x[key],
                               y=self.y[key],
                               a=self.a[key],
                               bx=self.bx[key],
                               by=self.by[key],
                               bz=self.bz[key],
                               time=self.time[key],
                               is_base_removed=self.is_base_removed,
                               date=self.date)

        if len(scan) == 1:
            scan.sampling_rate = None

        return scan
