import copy
import datetime
import pytest

import numpy as np
import matplotlib.pyplot as plt

from mag_utils.scans.base_scan import BaseScan
from mag_utils.scans.horizontal_scan import HorizontalScan


@pytest.fixture
def scan():
    return HorizontalScan(file_name="", x=[1.4, 2.2, 3.1], y=[1, 2, 3], a=[1, 2, 3], b=[1, 2, 3],
                          time=[datetime.time(hour=8, minute=4, second=53, microsecond=200000),
                                datetime.time(hour=8, minute=4, second=53, microsecond=300000),
                                datetime.time(hour=8, minute=4, second=53, microsecond=400000)]
                          , is_base_removed=False)


@pytest.fixture
def other_scan():
    return HorizontalScan(file_name="", x=[1.3], y=[2], a=[2], b=[2],
                          time=[datetime.time(hour=8, minute=4, second=53, microsecond=200000)], is_base_removed=False)


def test_z_property(scan: HorizontalScan):
    np.testing.assert_equal(scan.z, scan.a)


def test_mag_scan_init(scan):
    assert type(scan.x) is np.ndarray, f"Expected x to be np.ndarray but got {type(scan.x)}"
    assert type(scan.y) is np.ndarray, f"Expected y to be np.ndarray but got {type(scan.y)}"
    assert type(scan.a) is np.ndarray, f"Expected a to be np.ndarray but got {type(scan.a)}"
    assert type(scan.b) is np.ndarray, f"Expected b to be np.ndarray but got {type(scan.b)}"
    assert type(scan.time) is np.ndarray, f"Expected time to be np.ndarray but got {type(scan.time)}"
    assert scan.interpolated_data is None


def test_mag_scan_slicing(scan):
    expected_sliced_scan = HorizontalScan(file_name="", x=[1.4, 2.2], y=[1, 2], a=[1, 2], b=[1, 2],
                                          time=[datetime.time(hour=8, minute=4, second=53, microsecond=200000),
                                                datetime.time(hour=8, minute=4, second=53, microsecond=300000)],
                                          is_base_removed=False)

    for key, value in scan[0:2].__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_sliced_scan, key))
        else:
            assert value == getattr(expected_sliced_scan, key)


def test_mag_scan_appending(scan, other_scan):
    scan = scan.append(other_scan)
    expected_appended_scan = HorizontalScan(file_name="", x=[1.4, 2.2, 3.1, 1.3], y=[1, 2, 3, 2], a=[1, 2, 3, 2],
                                            b=[1, 2, 3, 2],
                                            time=[datetime.time(hour=8, minute=4, second=53, microsecond=200000),
                                                  datetime.time(hour=8, minute=4, second=53, microsecond=300000),
                                                  datetime.time(hour=8, minute=4, second=53, microsecond=400000),
                                                  datetime.time(hour=8, minute=4, second=53, microsecond=200000)],
                                            is_base_removed=False)

    for key, value in scan.__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_appended_scan, key))
        else:
            assert value == getattr(expected_appended_scan, key)


def test_mag_scan_slicing_mask(scan):
    expected_sliced_scan = HorizontalScan(file_name="", x=[2.2], y=[2], a=[2], b=[2],
                                          time=[datetime.time(hour=8, minute=4, second=53, microsecond=300000)],
                                          is_base_removed=False)

    mask = np.array([False, True, False])

    for key, value in scan[mask].__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_sliced_scan, key))
        else:
            assert value == getattr(expected_sliced_scan, key)


def test_mag_scan_plot(scan: HorizontalScan):
    _, ax = plt.subplots()
    assert scan.plot(ax) is not None, f"When ax is given to the plot method, return the output of the plot"
    assert scan.plot() is None, f"When nothing is given to the plot method, return None"


def test_mag_scan_colorbar_plot(scan: HorizontalScan):
    _, ax = plt.subplots()

    scatter_out = scan.plot(ax, colorbar=True)
    assert scatter_out.colorbar is not None, "The colorbar is displayed although it should not."

    scatter_out = scan.plot(ax, colorbar=False)
    assert scatter_out.colorbar is None, "The colorbar is not displayed although it should be."


def test_equals_objects():
    first_scan = HorizontalScan(file_name="scan.txt", x=[1.4, 2, 3.1], y=[1, 2, 3], a=[1, 2.4, 3.5], b=[1.9, 3.7, 5],
                                time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                               datetime.time(hour=8, minute=26, second=53, microsecond=550000)]),
                                is_base_removed=False)
    second_scan = copy.deepcopy(first_scan)

    # Case of compering equals scans
    assert first_scan == second_scan, f"The scans are equals, but received that they are not"

    # Case of compering different scans
    second_scan.b[0] = 5
    assert first_scan != second_scan, f"The scans are not equals, but received that they are"

    # Case of comparing objects that not of the same type
    with pytest.raises(TypeError) as e_info:
        first_scan == np.array([1.4, 2, 3.1])


def test_subtract_base_overloading(scan: HorizontalScan):
    base = BaseScan(file_name="", b=scan.b, time=scan.time)
    subtracted_scan = scan.subtract_base(base)

    np.testing.assert_array_equal(subtracted_scan.b, 0)
