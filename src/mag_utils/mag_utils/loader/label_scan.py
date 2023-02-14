"""Functions for loading a labeled mag scan."""
import h5py

from mag_utils.mag_utils.scans.labeled_scan import LabeledScan, Target
from mag_utils.mag_utils.loader import blackwidow


def h5_loader(path: str) -> LabeledScan:
    """
    Load labeled scan scan h5 file into LabeledScan object.

    Args:
        path: full path of labeled scan h5 file.

    Returns:
        LabeledScan object with the data of the file.
    """
    mag_scan_part = blackwidow.load(path)

    with h5py.File(path, 'r') as h5f:
        targets_group = h5f.get('targets')
        targets = []
        for target_group in targets_group.values():
            curr_target = Target.from_h5_group(target_group)
            targets.append(curr_target)

        is_real = h5f.attrs.get('is_real')
        sensor_type = h5f.attrs.get('sensor_type')

    # Create LabeledScan object with the data from the h5 file.
    scan = LabeledScan(file_name=mag_scan_part.file_name,
                       x=mag_scan_part.x,
                       y=mag_scan_part.y,
                       a=mag_scan_part.a,
                       b=mag_scan_part.b,
                       time=mag_scan_part.time,
                       date=mag_scan_part.date,
                       is_base_removed=mag_scan_part.is_base_removed,
                       sensor_type=sensor_type,
                       interpolated_data=mag_scan_part.interpolated_data,
                       is_real=is_real,
                       targets=targets)

    return scan


def load(path: str) -> LabeledScan:
    """
    Load a labeled blackwidow scan file into LabeledScan object.

    Supported file types - h5

    Args:
        path: full path of mag scan file.

    Returns:
        LabeledScan object with the data of the file.
    """
    if path.lower().endswith('.h5') or path.lower().endswith('.hdf5'):
        scan = h5_loader(path)
    else:
        raise ValueError(f'Unsupported file type.'
                         f' Only supporting h5 files.'
                         f' file: {path}')

    return scan
