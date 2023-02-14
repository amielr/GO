import datetime
import os

import pytest
import pandas as pd
import numpy as np

from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.saver.save_as_h5 import save


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


def test_removing_file_in_case_of_error(scan_mock):
    scan_mock.a = pd.DataFrame()
    output_path = 'test_removing_file_in_case_of_error.h5'

    with pytest.raises(TypeError):
        save(scan_mock, output_path)

    assert not os.path.exists(output_path)
