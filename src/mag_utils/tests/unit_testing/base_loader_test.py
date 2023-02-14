import os

import pytest
import datetime

import numpy as np

from mag_utils.loader import base
from mag_utils.scans.base_scan import BaseScan

mock_scan = BaseScan(file_name="base.txt", b=[1.9, 3.7, 5, 9.6],
                     time=np.array([datetime.time(hour=23, minute=4, second=53, microsecond=200000),
                                    datetime.time(hour=8, minute=26, second=53, microsecond=550000)]))


def test_load_txt():
    path = "../Data/base scans/after validation/Base_8_1_21_validated.tt"

    with pytest.raises(ValueError) as e_info:
        base.load(path)


def test_format_time():
    time = "19:11:06.055"
    expected_return = datetime.time(hour=19,
                                    minute=11,
                                    second=6,
                                    microsecond=55000)

    assert expected_return == base.format_time(time)
