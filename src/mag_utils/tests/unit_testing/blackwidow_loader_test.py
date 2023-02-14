import os

import pytest
import datetime

import numpy as np

from mag_utils.loader import blackwidow
from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.scans.interpolated_scan import InterpolatedScan


@pytest.fixture
def scan_mock():
    n = 10

    return HorizontalScan(file_name="scan.txt",
                          x=np.ones(n),
                          y=np.ones(n),
                          a=np.ones(n),
                          b=np.ones(n),
                          time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                         datetime.time(hour=8, minute=26, second=53, microsecond=550000)]),
                          is_base_removed=False,
                          date=datetime.date(year=1, month=1, day=1))


def test_load():
    path = "../Data/blackwidow scans/after validation/11_thirdspiral_validated.TXTt"

    with pytest.raises(ValueError) as e_info:
        blackwidow.load(path)


def test_format_time():
    time = "080453.20"
    expected_return = datetime.time(hour=8,
                                    minute=4,
                                    second=53,
                                    microsecond=200000)
    assert expected_return == blackwidow.format_time(time)

    time = "01:04:03.400000"
    expected_return = datetime.time(hour=1,
                                    minute=4,
                                    second=3,
                                    microsecond=400000)
    assert expected_return == blackwidow.format_time(time)


def test_save_and_load_h5(scan_mock):
    h5_path = f"widow_scan.h5"
    try:
        scan_mock.save(h5_path)

        loaded_scan = blackwidow.load(h5_path)

        assert loaded_scan == scan_mock
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e


def test_save_and_load_csv(scan_mock):
    csv_path = f"widow_scan.csv"
    try:
        scan_mock.save(csv_path)

        loaded_scan = blackwidow.load(csv_path)

        assert loaded_scan == scan_mock
        os.remove(csv_path)

    except Exception as e:
        os.remove(csv_path)
        raise e


def test_save_and_load_h5_scan_with_interpolation(scan_mock):
    n = 10
    scan_mock.interpolated_data = InterpolatedScan(x=np.ones([n, n]),
                                                   y=np.ones([n, n]),
                                                   b=np.ones([n, n]),
                                                   mask=np.ones([n, n], dtype=bool),
                                                   interpolation_method="")

    h5_path = f"widow_scan_interp.h5"
    try:
        scan_mock.save(h5_path)

        loaded_scan = blackwidow.load(h5_path)

        assert loaded_scan == scan_mock
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e


def test_save_load_save_h5_routine(scan_mock):
    file_path = 'tmp.h5'
    try:
        scan_mock.save(file_path)
        loaded_scan = blackwidow.load(file_path)
        loaded_scan.save(file_path)
        os.remove(file_path)

    except Exception as e:
        os.remove(file_path)
        raise e


def test_save_load_save_csv_routine(scan_mock):
    file_path = 'tmp.csv'
    try:
        scan_mock.save(file_path)
        loaded_scan = blackwidow.load(file_path)
        loaded_scan.save(file_path)
        os.remove(file_path)
    except Exception as e:
        os.remove(file_path)
        raise e
