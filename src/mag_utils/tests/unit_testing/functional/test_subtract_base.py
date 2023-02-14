import datetime

import numpy as np
import pandas as pd
import pytest

from mag_utils.functional.subtract_base import subtract_base
from mag_utils.functional.time import time2datetime


def get_end_of_day_in_seconds():
    return 60 * 60 * 24


def get_time_dummy(start, *, end=None, num_of_samples=None):
    """
    get datetime.time object with jumps of 100 ms

    Args:
        start: start time
        end: end time
        num_of_samples: instead of end time, specify how many samples you want from start.

    Returns: np.array(datetime.time)
    """
    if end is not None:
        timedeltas = pd.to_timedelta(np.arange(start, end, 0.1), unit='s')
    elif num_of_samples is not None:
        timedeltas = pd.to_timedelta(np.arange(start, start + 0.1 * num_of_samples, 0.1), unit='s')
    else:
        raise ValueError('either end or num_of_samples should be entered.')
    dummy_pd = timedeltas + pd.to_datetime('1/1/1970')
    dummy_datetime = dummy_pd.to_pydatetime()
    return np.array([t.time() for t in dummy_datetime])


def test_subtruct_base():
    scan_b, scan_t = np.arange(10), get_time_dummy(0.1, num_of_samples=10)
    base_b, base_t = np.arange(20), get_time_dummy(0, num_of_samples=20)
    subtracted_b = subtract_base(scan_b, scan_t, base_b, base_t)

    assert scan_b.shape == subtracted_b.shape
    np.testing.assert_equal(np.zeros_like(scan_b) - 1, subtracted_b)


def test_subtract_base_no_time_overlap():
    scan_b, scan_t = np.arange(10), get_time_dummy(2.3, num_of_samples=10)
    base_b, base_t = np.arange(20), get_time_dummy(get_end_of_day_in_seconds() - 1, num_of_samples=20)

    with pytest.raises(ValueError):
        subtract_base(scan_b, scan_t, base_b, base_t)


def test_base_starts_a_day_before_scan():
    scan_b, scan_t = np.arange(10), get_time_dummy(0, num_of_samples=10)
    base_b, base_t = np.arange(20), get_time_dummy(0, num_of_samples=20)

    dummy_datetime = np.asarray(time2datetime(base_t)) - datetime.timedelta(seconds=0.1)
    base_t = [t.time() for t in dummy_datetime]

    subtracted_b = subtract_base(scan_b, scan_t, base_b, base_t)
    np.testing.assert_equal(np.zeros_like(scan_b) - 1, subtracted_b)
