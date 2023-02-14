import copy
import datetime

import numpy as np
import pytest

from mag_utils.scans.aluminum_man import AluminumManScan


@pytest.fixture
def scan():
    bx = np.array([1.9, 3.7, 5])
    by = np.array([1.9, 3.7, 5])
    bz = np.array([1.9, 3.7, 5])

    return AluminumManScan(file_name="scan.txt", x=[1.4, 2, 3.1], y=[1, 2, 3], a=[1, 2.4, 3.5],
                           bx=bx,
                           by=by,
                           bz=bz,
                           time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                       datetime.time(hour=8, minute=26, second=53, microsecond=550000),
                                       datetime.time(hour=8, minute=26, second=59, microsecond=550000)]),
                           is_base_removed=False)


def test_equals_objects(scan):
    second_scan = copy.deepcopy(scan)

    # Case of compering equals scans
    assert scan == second_scan, f"The scans are equals, but received that they are not"

    # Case of compering different scans
    second_scan.b[0] = 5
    assert scan != second_scan, f"The scans are not equals, but received that they are"

    # Case of comparing objects that not of the same type
    with pytest.raises(TypeError) as e_info:
        scan == np.array([1.4, 2, 3.1])


def test_mag_scan_slicing_mask(scan):
    expected_sliced_scan = AluminumManScan(file_name="scan.txt", x=[2], y=[2], a=[2.4], bx=[3.7], by=[3.7], bz=[3.7],
                                           time=[datetime.time(hour=8, minute=26, second=53, microsecond=550000)],
                                           is_base_removed=False)

    mask = np.array([False, True, False])

    for key, value in scan[mask].__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_sliced_scan, key))
        else:
            assert value == getattr(expected_sliced_scan, key)
