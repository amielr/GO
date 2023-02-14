import matplotlib.pyplot as plt
from tkinter import filedialog
from pathlib import Path
from osgeo import gdal
import numpy as np
import warnings

def load_tiff(tiff_path: Path):

    ds = gdal.Open(r'{}'.format(tiff_path))
    gt = ds.GetGeoTransform()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    data = ds.ReadAsArray()
    data = np.ma.masked_values(data, nodata)

    return data, gt


def add_tiff_to_ax(tiff_ax: plt.axes, tiff_path: Path):
    """

    Args:
        tiff_ax: ax to plot on
        tiff_path: Tiff Path

    Returns:

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data, gt = load_tiff(tiff_path)

    xs = data.shape[2]
    ys = data.shape[1]
    ulx, x_res, _, uly, _, y_res = gt
    extent = [ulx, ulx + x_res * xs, uly + y_res * ys, uly]

    channels = np.dstack([data[0], data[1], data[2]])

    whites = np.array(channels != np.array([255, 255, 255])).any(axis=-1).astype(int) * 254
    channels = np.dstack([data[0], data[1], data[2], whites])
    tiff_ax.imshow(channels, extent=extent)

    ax.set_ylim(ax.axes.dataLim.ymin, ax.axes.dataLim.ymax)
    ax.set_xlim(ax.axes.dataLim.xmin, ax.axes.dataLim.xmax)
    return ax


if __name__ == '__main__':
    paths = [Path(path) for path in filedialog.askopenfilenames()]
    fig, ax = plt.subplots()
    for path in paths:
        add_tiff_to_ax(ax, path)
    plt.show()
