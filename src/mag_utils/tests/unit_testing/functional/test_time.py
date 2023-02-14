import datetime

import numpy as np
import pytest

from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.functional.time import compute_sampling_rate, find_midnight, time2datetime, times2seconds, \
    _is_just_before_midnight, find_time_jumpback


@pytest.fixture
def scan():
    return HorizontalScan(file_name="",
                          x=[1.4, 2.2, 3.1, 1.2],
                          y=[1, 2, 3, 4],
                          a=[1, 2, 3, 4],
                          b=[1, 2, 3, 4],
                          time=[datetime.time(hour=8, minute=4, second=53, microsecond=200000),
                                datetime.time(hour=8, minute=4, second=53, microsecond=300000),
                                datetime.time(hour=8, minute=4, second=53, microsecond=400000),
                                datetime.time(hour=8, minute=4, second=53, microsecond=500000)],
                          is_base_removed=False)


@pytest.fixture
def times():
    return [
        datetime.time(hour=23, minute=59, second=59, microsecond=600000),
        datetime.time(hour=23, minute=59, second=59, microsecond=700000),
        datetime.time(hour=23, minute=59, second=59, microsecond=800000),
        datetime.time(hour=23, minute=59, second=59, microsecond=900000),
        datetime.time(hour=0, minute=0, second=0, microsecond=000000),
        datetime.time(hour=0, minute=0, second=0, microsecond=100000),
        datetime.time(hour=0, minute=0, second=0, microsecond=200000),
        datetime.time(hour=0, minute=0, second=0, microsecond=300000)
    ]


def test_sampling_rate_computation(scan: HorizontalScan):
    assert compute_sampling_rate(scan.time) == 10


def test_down_sampling_scan(scan):
    down_sampled_scan = scan[::2]
    assert down_sampled_scan.sampling_rate == 5


def test_sampling_rate_computation_at_midnight(times):
    assert compute_sampling_rate(times) == 10


def test_downsampling_single_time_point():
    times = [datetime.time(hour=23, minute=59, second=59, microsecond=900000)]

    assert compute_sampling_rate(times) is None


def test_downsampling_with_missing_time_vec(scan):
    scan = HorizontalScan(file_name="",
                          x=[1.4, 2.2, 3.1, 1.2],
                          y=[1, 2, 3, 4],
                          a=[1, 2, 3, 4],
                          b=[1, 2, 3, 4],
                          time=None,
                          is_base_removed=False)

    assert scan.sampling_rate is None


def test_find_time_jumpback(times):
    times.extend(times)
    np.testing.assert_equal(find_time_jumpback(times), [3, 11])


def test__is_just_before_midnight():
    assert _is_just_before_midnight(datetime.time(23, 59, 0))
    assert not _is_just_before_midnight(datetime.time(23, 54, 0))
    assert _is_just_before_midnight(datetime.time(23, 54, 0), tolerance=datetime.timedelta(minutes=6))


def test_find_midnight(times):
    assert find_midnight(times) == 3
    assert find_midnight(times[:4]) is None
    assert find_midnight(times[3:]) == 0


def test_time2datetime(times):
    date1 = datetime.date(1997, 1, 1)
    date2 = datetime.date(1997, 1, 2)
    time_with_dates = time2datetime(times, date1)
    assert time_with_dates[0].date() == date1
    assert time_with_dates[-1].date() == date2


def test_times2seconds(times):
    np.testing.assert_almost_equal(times2seconds(times), np.arange(0, len(times) * 0.1, 0.1))
