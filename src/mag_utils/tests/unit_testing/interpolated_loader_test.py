import os

import pytest

import numpy as np

from mag_utils.loader import interpolated
from mag_utils.scans.interpolated_scan import InterpolatedScan


@pytest.fixture()
def mock_scan():
    arr = np.ones([2, 3])

    return InterpolatedScan(x=arr, y=arr, b=arr, mask=arr, interpolation_method="RBF")


def test_save_and_load_tiff(mock_scan):
    tiff_path = "interpolated_scan.tif"
    mock_scan.save(tiff_path)

    loaded_scan = interpolated.load(tiff_path)

    assert loaded_scan == mock_scan

    os.remove(tiff_path)


def test_save_and_load_json(mock_scan):
    json_path = "interpolated_scan.json"
    mock_scan.save(json_path)

    loaded_scan = interpolated.load(json_path)

    assert loaded_scan == mock_scan

    os.remove(json_path)
