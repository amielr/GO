import copy
import pytest

import numpy as np
import matplotlib.pyplot as plt

from mag_utils.scans.interpolated_scan import InterpolatedScan


@pytest.fixture
def scan():
    n = 10
    return InterpolatedScan(x=np.ones([n, n]),
                            y=np.ones([n, n]),
                            b=np.ones([n, n]),
                            mask=np.ones([n, n], dtype=bool),
                            interpolation_method="")


def test_interpolated_scan_init():
    scan = InterpolatedScan(x=[1.4, 2.2, 3.1], y=np.array([[1, 2, 3], [1, 4, 7]]), b=[],
                            mask=np.array([[1, 2, 3], [1, 4, 7]]), interpolation_method="")

    assert type(scan.x) is np.ndarray, f"Expected x to be np.ndarray but got {type(scan.x)}"
    assert type(scan.y) is np.ndarray, f"Expected y to be np.ndarray but got {type(scan.y)}"
    assert type(scan.b) is np.ndarray, f"Expected b to be np.ndarray but got {type(scan.b)}"
    assert type(scan.mask) is np.ndarray, f"Expected b to be np.ndarray but got {type(scan.mask)}"


def test_plot(scan):
    _, ax = plt.subplots()
    assert scan.plot(ax=ax) is not None, f"When ax is given to the plot method, return the output of the plot"
    assert scan.plot() is None, f"When nothing is given to the plot method, return None"


def test_interpolated_scan_colorbar_plot(scan: InterpolatedScan):
    _, ax = plt.subplots()

    cont_out = scan.plot(ax=ax, colorbar=True)
    assert cont_out.colorbar is not None, "The interpolated scan plot does not have a colorbar although it should."

    cont_out = scan.plot(ax=ax, colorbar=False)
    assert cont_out.colorbar is None, "The interpolated scan plot has a colorbar although it should."


def test_equals_objects():
    first_scan = InterpolatedScan(x=[1.4, 2.2, 3.1], y=np.array([[1, 2, 3], [1, 4, 7]]), b=[],
                                  mask=np.array([[1, 2, 3], [1, 4, 7]]), interpolation_method="")
    second_scan = copy.deepcopy(first_scan)

    # Case of compering equals scans
    assert first_scan == second_scan, f"The scans are equals, but received that they are not"

    # Case of compering different scans
    second_scan.mask[0][0] = 5
    assert first_scan != second_scan, f"The scans are not equals, but received that they are"

    # Case of comparing objects that not of the same type
    with pytest.raises(TypeError) as e_info:
        first_scan == np.array([1.4, 2, 3.1])
