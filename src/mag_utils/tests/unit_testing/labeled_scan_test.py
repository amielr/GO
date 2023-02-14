import datetime

import pytest
import numpy as np
from mag_utils.labeled_scan import LabeledScan, Target


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
                             datetime.time(hour=8, minute=26, second=53, microsecond=450000),
                             datetime.time(hour=8, minute=26, second=53, microsecond=550000)],
                       sensor_type='widow',
                       is_base_removed=False,
                       targets=targets,
                       is_real=False)

    return scan


def test_labeled_scan_slicing(mock_scan, targets):
    expected_sliced_scan = LabeledScan(file_name="mock_scan",
                                       x=[1.4, 2.2],
                                       y=[1, 2],
                                       a=[1, 2],
                                       b=[1, 2],
                                       time=[datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                             datetime.time(hour=8, minute=26, second=53, microsecond=450000)],
                                       sensor_type='widow',
                                       is_base_removed=False,
                                       targets=targets,
                                       is_real=False)

    # assert output type is LabeledScan
    assert isinstance(mock_scan[0:2], LabeledScan)

    for key, value in mock_scan[0:2].__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_equal(value, getattr(expected_sliced_scan, key))
        else:
            assert value == getattr(expected_sliced_scan, key)
