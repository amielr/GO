import numpy as np

from mag_utils.scans.mag_scan import HorizontalScan
from datetime import datetime, timedelta
from pytest import fixture
from mag_utils.functional.line_analysis import separate_lines


@fixture
def mock_line_scan():
    single_line_x = np.arange(0, 100, 0.1)
    x = np.concatenate([single_line_x, np.flip(single_line_x)])

    MOCK_DATETIME = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    return HorizontalScan(file_name="mock_scan",
                          x=x,
                          y=np.ones_like(x),
                          b=np.ones_like(x),
                          a=np.ones_like(x),
                          time=[(MOCK_DATETIME + timedelta(seconds=t / 10)).time() for t in np.arange(len(x))],
                          is_base_removed=False)


def test_separate_lines(mock_line_scan):
    volvoed_scan, lines = separate_lines(mock_line_scan, cluster_distance=1, min_cluster_size=3, time_resolution=5,
                                         perpendicular_state_flag=True, angle_tolerance_value=60,
                                         opposite_state_flag=True, mean_by_dir_flag=False,
                                         substract_mean_per_line_flag=False)
    assert len(lines) == 2, "The VOLVO missed some lines"
