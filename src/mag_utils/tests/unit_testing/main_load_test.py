import os

import mock
import pytest
import datetime

import numpy as np

from mag_utils.scans.aluminum_man import AluminumManScan
from mag_utils.scans.base_scan import BaseScan
from mag_utils.scans.labeled_scan import LabeledScan, Target
from mag_utils import load
from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.scans.interpolated_scan import InterpolatedScan


@pytest.fixture
def mock_aluminum_file_text():
    return """VERSION 20F1 : 1.2.05 Frequency : 25.00 Hz	Filter : 36
[CHANNEL] [NUMERO_TRAME] [INDICE_GPS] [FLAG] [TEMPERATURE1] [X1] [Y1] [Z1]
>G01 GNGGA,122144.00,3141.76088,N,03440.45308,E,1,08,1.35,63.3,M,16.8,M,,*7B
>M01 00000000 00000000 0 +21.7 +0211666 +0211004 +0219592
>M01 00000001 00000000 0 +21.7 +0211594 +0210920 +0219518
>M01 00000002 00000000 0 +21.7 +0210658 +0209992 +0218586
>M01 00000003 00000000 0 +21.7 +0367780 -1819270 -1809632"""


@pytest.fixture
def targets():
    return [Target("plate", [23, 1, 3], [1, 2, 3]),
            Target("rocket", [1, 2, 4], [1, 2, 1])]


@pytest.fixture
def labeled_scan_mock(targets):
    scan = LabeledScan(file_name="labeled_scan.txt",
                       x=[1.4, 2.2, 3.1],
                       y=[1, 2, 3],
                       a=[1, 2, 3],
                       b=[1, 2, 3],
                       time=[datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                             datetime.time(hour=8, minute=26, second=53, microsecond=550000)],
                       sensor_type='widow',
                       is_base_removed=False,
                       targets=targets,
                       is_real=False)

    return scan


@pytest.fixture
def blackwidow_scan_mock():
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

@pytest.fixture
def aluminum_scan_mock():
    n = 10

    return AluminumManScan(file_name="aluminum_scan.txt",
                           x=np.ones(n),
                           y=np.ones(n),
                           a=np.ones(n),
                           bx=np.ones(n),
                           by=np.ones(n),
                           bz=np.ones(n),
                           time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                          datetime.time(hour=8, minute=26, second=53, microsecond=550000)]),
                           is_base_removed=False,
                           date=datetime.date(year=1, month=1, day=1))


@pytest.fixture
def base_scan_mock():
    base_scan = BaseScan(file_name="base.txt", b=[1.9, 3.7, 5, 9.6],
                         time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                        datetime.time(hour=23, minute=4, second=53, microsecond=30000),
                                        datetime.time(hour=23, minute=4, second=53, microsecond=400000),
                                        datetime.time(hour=23, minute=4, second=53, microsecond=500000)]))

    return base_scan


@pytest.fixture()
def interpolated_scan_mock():
    arr = np.ones([2, 3])

    return InterpolatedScan(x=arr, y=arr, b=arr, mask=arr, interpolation_method="RBF")


def test_save_and_load_h5_widow(blackwidow_scan_mock):
    h5_path = r"widow_scan.h5"
    try:
        blackwidow_scan_mock.save(h5_path)

        loaded_scan = load(h5_path)

        assert loaded_scan == blackwidow_scan_mock
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e


def test_save_and_load_h5_base(base_scan_mock):
    h5_path = r"base_scan.h5"
    try:
        base_scan_mock.save(h5_path)

        loaded_scan = load(h5_path)

        assert loaded_scan == base_scan_mock
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e


def test_save_and_load_csv_widow(blackwidow_scan_mock):
    csv_path = r"widow_scan.csv"
    try:
        blackwidow_scan_mock.save(csv_path)

        loaded_scan = load(csv_path)

        assert loaded_scan == blackwidow_scan_mock
        os.remove(csv_path)

    except Exception as e:
        os.remove(csv_path)
        raise e


def test_save_and_load_json_interpolated(interpolated_scan_mock):
    json_path = "interpolated_scan.json"
    interpolated_scan_mock.save(json_path)

    loaded_scan = load(json_path)

    assert loaded_scan == interpolated_scan_mock

    os.remove(json_path)


def test_save_and_load_tiff_interpolated(interpolated_scan_mock):
    json_path = "interpolated_scan.tif"
    interpolated_scan_mock.save(json_path)

    loaded_scan = load(json_path)

    assert loaded_scan == interpolated_scan_mock

    os.remove(json_path)


def test_save_and_load_h5_labeled_scan(labeled_scan_mock):
    h5_path = f"labeled_scan.h5"
    try:
        labeled_scan_mock.save(h5_path)

        loaded_scan = load(h5_path)

        assert loaded_scan == labeled_scan_mock, "scan is not the same."
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e


def test_save_and_load_txt_widow_validation(blackwidow_scan_mock):
    txt_path = r"widow_scan.txt"

    text_widow_with_wrong_line_mock = \
        "$GPGGA,090110.20,3120.1056930,N,03422.5723738,E,2,10,0.8,104.478,M,17.80,M,18,TSTR*55, 44747.678,1467\n" \
        "$GPGGA,090110.30,3120.1056935,N,03422.5723736,E,2,10,0.8,104.479,M,17.80,M,18,TSTR*5E, 44747.638,1470\n" \
        "$GasdaPGGA,090109.90,3120.1056938,N,03422.5723730,E,2,1,M,17.80,M,1 sdfs8,TSTR*59, 44747.711,1470\n" \
        "$GaPsadGGA,090110.00,3120.1056922,N,03422.5723749,sdfsfE,2,17fgdg.80,M,18,TSTR*5D, 44747.488,1470\n" \
        "$GasdPGGA,09sad0110.10,sad.10sad56942,asdN,03422.5723743sdfsdE,,17.80,M,18,dsf*5F, 44747.655,1467\n" \
        "$GPGGA,090110.40,3120.1056940,N,03422.5723738,E,2,10,0.8,104.480,M,17.80,M,18,TSTR*53, 44747.605,1467\n" \
        "$GPGGA,090110.50,3120.1056924,N,03422.5723745,E,2,10,0.8,104.482,M,17.80,M,19,TSTR*59, 44747.676,1467\n"

    try:
        with open(txt_path, "w") as file:
            file.writelines(text_widow_with_wrong_line_mock)

        scan = load(txt_path, validate=True, save_validated_file=False)

        assert isinstance(scan, HorizontalScan)

        os.remove(txt_path)
    except Exception as e:
        os.remove(txt_path)
        raise e


def test_save_and_load_h5_aluminum_scan(aluminum_scan_mock):
    h5_path = f"aluminum_scan.h5"
    try:
        aluminum_scan_mock.save(h5_path)

        aluminum_scan = load(h5_path)

        assert aluminum_scan == aluminum_scan_mock, "scan is not the same."
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e


def test_load_aluminum_txt_file(mock_aluminum_file_text):
    txt_path = "file.txt"
    try:
        with open(txt_path, "w") as file:
            file.writelines(mock_aluminum_file_text)

        scan = load(txt_path)

        assert isinstance(scan, AluminumManScan)

        os.remove(txt_path)
    except Exception as e:
        os.remove(txt_path)
        raise e
