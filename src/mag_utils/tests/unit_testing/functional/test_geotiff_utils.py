import matplotlib.pyplot as plt

from mag_utils.functional.geotiff_utils import create_tiff, create_png
from ..labeled_scan_loader_test import mock_mag_scan
from pathlib import Path


def test_create_tiff(mock_mag_scan):
    save_path = Path("geotiff.tif")

    # create mock scan and plot it
    _, ax = plt.subplots(1)
    mock_mag_scan.plot(ax)

    # create geotiff
    create_tiff(ax, str(save_path))

    # ensure created
    tiff_exists = save_path.exists()

    if tiff_exists:
        save_path.unlink()

    assert tiff_exists, "Could not find geotiff"


def test_create_png(mock_mag_scan):
    # create mock scan and plot it
    _, ax = plt.subplots(1)
    mock_mag_scan.plot(ax)

    img_array = create_png(ax)

    assert len(img_array), "PNG array is invalid"


