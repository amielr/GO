import datetime

import numpy as np

from mag_utils.cleaners.magen import auto_magen_cleaning
from mag_utils.scans import HorizontalScan
import pytest


@pytest.fixture
def scan_mock():
    n = 50

    return HorizontalScan(file_name="scan.txt",
                          x=np.random.randn(n) * 2,
                          y=np.random.randn(n) * 2,
                          a=np.random.randn(n) * 2,
                          b=np.zeros(n),
                          time=np.array(
                              [datetime.time(hour=0, minute=0, second=0, microsecond=0)
                               for _ in range(n)]),
                          is_base_removed=False,
                          date=datetime.date(year=1, month=1, day=1))


def test_auto_magen_cleaning(scan_mock):
    clean_scan = auto_magen_cleaning(scan_mock,
                                     dipole_spacing=4,
                                     depth_search_interval=(10, 10))

    assert isinstance(clean_scan, HorizontalScan)
