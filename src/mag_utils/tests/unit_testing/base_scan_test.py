import copy
import datetime

import pytest

from mag_utils.scans.base_scan import BaseScan
import numpy as np


@pytest.fixture
def mock_base():
    base_scan = BaseScan(file_name="base.txt", b=[1.9, 3.7, 5, 9.6],
                         time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                        datetime.time(hour=23, minute=4, second=53, microsecond=30000),
                                        datetime.time(hour=23, minute=4, second=53, microsecond=400000),
                                        datetime.time(hour=23, minute=4, second=53, microsecond=500000)]))

    return base_scan


@pytest.fixture
def time_vector():
    times = np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                      datetime.time(hour=23, minute=4, second=53, microsecond=300000),
                      datetime.time(hour=23, minute=4, second=53, microsecond=400000)])

    return times


def test_base_scan_init(time_vector):
    scan = BaseScan(file_name="", b=[1, 2, 3], time=time_vector)

    assert type(scan.b) is np.ndarray, f"Expected b to be np.ndarray but got {type(scan.b)}"
    assert type(scan.time) is np.ndarray, f"Expected time to be np.ndarray but got {type(scan.time)}"


def test_base_scan_slicing(time_vector):
    original_scan = BaseScan(file_name="", b=[1, 2, 3], time=time_vector)

    expected_sliced_scan = BaseScan(file_name="", b=[1, 2], time=time_vector[:2])

    for key, value in original_scan[0:2].__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_sliced_scan, key))
        else:
            assert value == getattr(expected_sliced_scan, key)


def test_base_scan_slicing_mask(time_vector):
    original_scan = BaseScan(file_name="", b=[1, 2, 3], time=time_vector)

    expected_sliced_scan = BaseScan(file_name="", b=[2], time=[time_vector[1]])

    mask = np.array([False, True, False])

    for key, value in original_scan[mask].__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_sliced_scan, key))
        else:
            assert value == getattr(expected_sliced_scan, key)


def test_equals_objects():
    first_scan = BaseScan(file_name="base.txt", b=[1.9, 3.7, 5, 9.6],
                          time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                         datetime.time(hour=8, minute=26, second=53, microsecond=550000)]))
    second_scan = copy.deepcopy(first_scan)

    # Case of compering equals scans
    assert first_scan == second_scan, f"The scans are equals, but received that they are not"

    # Case of compering different scans
    second_scan.b[0] = 5
    assert first_scan != second_scan, f"The scans are not equals, but received that they are"

    # Case of comparing objects that not of the same type
    with pytest.raises(TypeError) as e_info:
        first_scan == np.array([1.4, 2, 3.1])
