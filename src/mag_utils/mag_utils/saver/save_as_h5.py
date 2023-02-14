"""Function that save object data in h5"""
import os

import h5py
import numpy as np


def save_arrays(h5_file, attr_name, attr_value):
    # Add time numpy array as string dataset.
    if attr_name in ["time", "date", "date_time"]:
        h5_file.create_dataset(attr_name, data=attr_value.astype(str).tolist())
    # Add numpy array as dataset.
    else:
        h5_file.create_dataset(attr_name, data=attr_value)


def save_targets(h5_file, targets):
    targets_group = h5_file.create_group("targets")
    for i, target in enumerate(targets):
        target_group = targets_group.create_group(f"target_{i}")
        target.to_h5_group(target_group)


def save_spacial_cases(h5_file, attr_name, attr_value):
    # Create group of interpolated_data and add it's data to it.
    if attr_name == "interpolated_data":
        group = h5_file.create_group(attr_name)
        save(scan=attr_value, opened_h5=group)

    elif attr_name == "targets":
        save_targets(h5_file, attr_value)

    # Add date as string attribute.
    elif attr_name == "date":
        h5_file.attrs.create(attr_name, str(attr_value))


def save(scan, output_path=None, opened_h5=None):
    """
    Save the data in h5 file in the following format:
    - numpy arrays as datasets.
    - time numpy array as dataset of strings.
    - int, bool, string, float as attribute
    - date as string attribute
    - interpolated_data as a group. If it is none the group will be empty, otherwise the data will be save according to
    the above patterns.
    all the names are the object attributes names.

    Args:
        scan: Object of magScanBase, InterpolatedScan, or BaseScan (Can work wint any other object).
        output_path: The output file path.
        opened_h5: Instance of opened h5 (can be instance of opened group instead)
    """
    if scan is None:
        return

    h5_file = h5py.File(output_path, 'w') if opened_h5 is None else opened_h5

    try:
        for attr_name, attr_value in scan.__dict__.items():
            if isinstance(attr_value, np.ndarray):
                save_arrays(h5_file, attr_name, attr_value)

            # Add int, bool, str or float as attribute.
            elif isinstance(attr_value, (int, bool, str, float)):
                h5_file.attrs.create(attr_name, attr_value)

            elif attr_name in ["interpolated_data", "targets", "date"]:
                save_spacial_cases(h5_file, attr_name, attr_value)

            else:
                raise TypeError(f"Don't have option of saving {attr_name} of type {type(attr_value)} in the h5")

        # In case we open instance of h5 (receive path and not opened_h5), so close the file instance.
        if opened_h5 is None:
            h5_file.close()

    except Exception as exp:
        h5_file.close()
        os.remove(output_path)
        raise exp
