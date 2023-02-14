import datetime

import numpy as np

from .time import time2datetime, times2seconds
from .._consts import Sequence


def _undo_sort(sorted_array: np.ndarray, idxs: Sequence):
    """
    Return a sorted array to it's original ordering.

    Args:
        sorted_array: A sorted array.
        idxs: The indices that sorted the original array. Usually the output of np.argsort.

    Returns:
        The array with its original ordering.
    """
    i = np.empty_like(idxs)
    i[idxs] = np.arange(len(idxs))
    array = sorted_array[i]

    return array

def subtract_base(scan_b: Sequence, scan_t, base_b, base_t) -> np.ndarray:
    """
    Subtract the base scan values (base_b) from the main scan (scan_b).

    Notes:
        assuming no scan is longer than 24 hours.

    Args:
        scan_b: value of scan points. (in float or whatever)
        scan_t: timestamps of each scan point - datetime.time
        base_b: values of base scan. (in float or whatever)
        base_t: timestamps of each base scan point. datetime.time

    Returns: subtracted b.
    """

    scan_t_with_date = np.asarray(time2datetime(scan_t))
    base_t_with_date = np.asarray(time2datetime(base_t))

    scan_t_sort_idxs = np.argsort(scan_t_with_date)
    base_t_sort_idxs = np.argsort(base_t_with_date)

    scan_t_with_date = scan_t_with_date[scan_t_sort_idxs]
    base_t_with_date = base_t_with_date[base_t_sort_idxs]

    # in case the base starts a day before the scan.
    if scan_t_with_date[0] < base_t_with_date[0]:
        scan_t_with_date += datetime.timedelta(days=1)

    if scan_t_with_date[0] < base_t_with_date[0] or scan_t_with_date[-1] > base_t_with_date[-1]:
        raise ValueError(f"""times mismatch.
scan first and last times {scan_t[0]} - {scan_t[-1]}.
base first and last times {base_t[0]} - {base_t[-1]}.""")

    first_elem_in_base_scan = np.argwhere(np.array(base_t_with_date) >= scan_t_with_date[0])[0][0]

    base_t_sec = times2seconds(np.asarray(base_t)[base_t_sort_idxs])
    scan_t_sec = times2seconds(np.asarray(scan_t)[scan_t_sort_idxs], offset=base_t_sec[first_elem_in_base_scan])

    interpolated_base_b_sorted = np.interp(x=scan_t_sec, xp=base_t_sec, fp=base_b)

    interpolated_base_b = _undo_sort(interpolated_base_b_sorted, scan_t_sort_idxs)

    subtracted_b = scan_b - interpolated_base_b

    return subtracted_b
