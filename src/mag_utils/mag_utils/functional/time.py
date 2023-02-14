"""Time manipulations functions."""

import datetime
from typing import List

import numpy as np
import pandas as pd

from .._consts import Sequence

MOCK_DATE = datetime.date(1970, 1, 1)


def format_time(time: str) -> datetime.time:
    """
    Converts a time-stamp string to datetime.time object.

    Supports iso formats, hhmmss.ss and hh:mm:ss.sss

    Args:
        time: time in format 'hhmmss.ml' or iso format.

    Returns:
        datatime.time object
    """
    try:
        return datetime.datetime.fromisoformat(time).time()
    except ValueError:
        time = time.replace(":", "")

        missing_h_format = 'hmmss.l'
        # if first h in hhmmss.ml is missing, add 0 to the start.
        if len(time) == len(missing_h_format) and time.index('.') == missing_h_format.index('.'):
            time = '0' + time

        try:
            ms_s = time[6:]
        except IndexError:
            ms_s = 0

        microsecond = int(float(ms_s) * 1_000_000) if ms_s else 0

        formatted_time = datetime.time(hour=int(time[:2]),
                                       minute=int(time[2:4]),
                                       second=int(time[4:6]),
                                       microsecond=microsecond)

        return formatted_time


def find_time_jumpback(times: List[datetime.time]) -> List[int]:
    """
    Get the indexes of just before where the timestamps jumps back in time.

    Useful to find midnight and concatenated scans.

    Args:
        times: the time stamps sequence

    Returns:
        A list of the indexes. Empty list if there's no jump.
    """
    time_with_date = [datetime.datetime.combine(MOCK_DATE, t) for t in times]
    deltas = pd.to_timedelta(np.diff(time_with_date))
    jumpback_idxs = np.nonzero(deltas.days == -1)[0]

    return jumpback_idxs.tolist()


def _is_just_before_midnight(timestamp: datetime.time, tolerance=datetime.timedelta(minutes=5)) -> bool:
    """
    Check if a given timestamp happened before midnight, with an adjustable tolerance .

    Args:
        timestamp: the time stamp to test
        tolerance: a timedelta of what counts as close. Default is 5 minutes.

    Returns:
        if the timestamp happened before midnight.

    """
    time_with_date = datetime.datetime.combine(MOCK_DATE, timestamp)
    midnight = datetime.datetime.combine(MOCK_DATE + datetime.timedelta(days=1), datetime.time(0, 0, 0))

    return midnight - time_with_date <= tolerance


def find_midnight(times: List[datetime.time]) -> int:
    """Get the index just before where the timestamps passes midnight. None if there is no day pass."""

    jumpback_idxs = find_time_jumpback(times)

    day_skip_idx = None
    for jumpback_idx in jumpback_idxs:
        curr_time = times[jumpback_idx]
        if _is_just_before_midnight(curr_time):
            day_skip_idx = jumpback_idx
            break

    return day_skip_idx


def time2datetime(times: List[datetime.time], date: datetime.date = None) -> List[datetime.datetime]:
    """
    Turn a timestamps list to a datetime list.

    If we pass midnight, the day will increase by one.

    Args:
        times: A list of datetime.time objects
        date: The date to append to the time stamps.

    Returns:
        A list of datetime.datetime objects.
    """
    date = MOCK_DATE if date is None else date
    time_with_date = np.array([datetime.datetime.combine(date, t) for t in times])

    midnight_idx = find_midnight(times)
    if midnight_idx is not None:
        time_with_date[midnight_idx + 1:] += datetime.timedelta(days=1)

    return time_with_date.tolist()


def times2seconds(times, offset: float = 0) -> np.ndarray:
    """
    Convert times stamps to seconds from start.

    Args:
        times: Time vector of datetime.time objects.
        offset: in seconds.

    Returns:
        A numpy array of the cumulative seconds.

    """
    time_with_date = time2datetime(times)
    deltas = pd.to_timedelta(np.diff(time_with_date))
    seconds = np.cumsum(deltas.total_seconds())
    seconds = np.append(0, np.asarray(seconds))

    return seconds + offset


def compute_sampling_rate(times: Sequence) -> [float, None]:
    """
    Computes the sampling rate given a time vector.

    Computes the sampling rate by calculating the median of the time differences.
    Since it uses the median, it can work with a time vector that has at most 50% missing samples.

    Args:
        times: Time vector of datetime.time objects.

    Returns:
        Sampling rate in Hz.

    """
    if len(times) == 1:
        return None

    time_with_date = time2datetime(times)
    deltas = np.diff(time_with_date)
    deltas_in_seconds = np.array([delta.microseconds / 1e6 for delta in deltas])
    period_time = np.median(deltas_in_seconds)

    return 1. / period_time


def convert_datetime_to_seconds(time: datetime.time):
    """
    Convert datetime to seconds.

    Args:
        time: time in datetime format

    Returns:
        number of seconds.
    """
    return (time.hour * 60 + time.minute) * 60 + time.second
