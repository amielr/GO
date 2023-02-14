from mag_utils.algo_tools.convex_hull_grids import rectangular_convex_hull_grid_at_depths
from pytest import fixture
import numpy as np
from mag_utils.scans import HorizontalScan
from datetime import datetime, timedelta


@fixture
def mock_lines_scan():
    single_line_x = np.arange(0, 100, 0.1)
    x = np.concatenate([single_line_x, np.flip(single_line_x)])

    # assemble y
    y = np.ones_like(x)
    y[len(y) // 2:] = 30

    MOCK_DATETIME = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    return HorizontalScan(file_name="mock_scan",
                          x=x,
                          y=y,
                          b=np.ones_like(x),
                          a=np.ones_like(x),
                          time=[(MOCK_DATETIME + timedelta(seconds=t / 10)).time() for t in np.arange(len(x))],
                          is_base_removed=False)


def test_rectangular_convex_hull_grid_at_depths(mock_lines_scan):
    x_values, y_values, z_values, n_major, n_minor = rectangular_convex_hull_grid_at_depths(mock_lines_scan.x,
                                                                                            mock_lines_scan.y,
                                                                                            depths=[0])

    assert len(x_values) \
           and len(y_values)\
           and len(z_values), "Rectangular convex hull is invalid"
