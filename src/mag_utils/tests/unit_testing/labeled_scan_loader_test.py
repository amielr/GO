import os
import datetime

import pytest

from mag_utils.scans.labeled_scan import LabeledScan, Target
from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.loader import label_scan


@pytest.fixture
def targets():
    return [Target("plate", [23, 1, 3], [1, 2, 3]),
            Target("rocket", [1, 2, 4], [1, 2, 1])]


@pytest.fixture
def mock_scan(targets):
    scan = LabeledScan(file_name="mock_scan",
                       x=[1.4, 2.2, 3.1],
                       y=[1, 2, 3],
                       a=[1, 2, 3],
                       b=[1, 2, 3],
                       time=[datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                             datetime.time(hour=8, minute=26, second=53, microsecond=550000)],
                       is_base_removed=False,
                       targets=targets,
                       sensor_type='widow',
                       is_real=False)

    return scan


@pytest.fixture
def mock_mag_scan():
    scan = HorizontalScan(file_name="mock_scan",
                          x=[1.4, 2.2, 3.1],
                          y=[1, 2, 3],
                          a=[1, 2, 3],
                          b=[1, 2, 3],
                          time=[datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                datetime.time(hour=8, minute=26, second=53, microsecond=550000)],
                          is_base_removed=False)

    return scan


def test_save_and_load_h5(mock_scan):
    h5_path = f"widow_scan.h5"
    try:
        mock_scan.save(h5_path)

        loaded_scan = label_scan.load(h5_path)

        assert loaded_scan == mock_scan, "scan is not the same."
        os.remove(h5_path)

    except Exception as e:
        os.remove(h5_path)
        raise e

def test_add_labels(mock_mag_scan, targets):
    labeled_scan = LabeledScan.label(mock_mag_scan, targets, is_real=True, sensor_type='widow')
    assert type(labeled_scan) == LabeledScan
    assert labeled_scan.targets == targets
