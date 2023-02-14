from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from osgeo import gdal, osr

GEO_PROJECTION = 32636


def create_png(ax, temp_png_dpi=600, final_dpi=200):
    """
    Args:
        ax: matplotlib axis that is to be exported
        temp_png_dpi: dpi of temporary png file
        final_dpi: dpi of final tiff file

    Returns: np.array that contains the image values

    """
    # get figure from axis
    fig = ax.get_figure()

    ax.axis("off")
    fig.set_dpi(temp_png_dpi)

    # save as png
    png1 = BytesIO()
    fig.savefig(png1, format="png")

    # open, then save as tiff
    png2 = Image.open(png1)

    temp_tif_path = Path.home() / "temp.tif"
    png2.save(str(temp_tif_path))
    png1.close()

    ax.axis("on")
    fig.set_dpi(final_dpi)

    # delete temp tiff file
    temp_tif_path.unlink(missing_ok=True)

    return np.array(png2)


def create_tiff(ax, save_path, temp_png_dpi=600, final_dpi=200):
    """
    Args:
        ax: matplotlib axis that is to be exported
        save_path: where the tiff is saved
        temp_png_dpi: dpi of temporary png file
        final_dpi: dpi of final tiff file

    Returns: None

    """
    # extract channels from image
    image = create_png(ax, temp_png_dpi=temp_png_dpi, final_dpi=final_dpi)
    red, green, blue, alpha = image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]

    # reduce extrapolation
    red, green, blue, alpha = reduce_borders(red, green, blue, alpha)

    # create geotransform
    geotransform, nrows, ncols = get_geotransform(ax, alpha)

    # apply geotransform to tiff

    geotif = gdal.GetDriverByName('GTiff').Create(save_path, ncols, nrows, 4, gdal.GDT_Byte)
    geotif.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(GEO_PROJECTION)
    geotif.SetProjection(srs.ExportToWkt())

    # write all channels to geotiff
    geotif.GetRasterBand(1).WriteArray(red)
    geotif.GetRasterBand(2).WriteArray(green)
    geotif.GetRasterBand(3).WriteArray(blue)
    geotif.GetRasterBand(4).WriteArray(alpha)
    geotif.FlushCache()


def reduce_borders(red, green, blue, alpha):
    """
    Cut whitespace that surrounds the image.

    Args:
        red: the red channel of the image
        green: the green channel of the image
        blue: the blue channel of the image
        alpha: the alpha channel of the image

    Returns: red, green, blue, alpha of the resulting image

    """
    x_borders, y_borders = np.where(alpha != 0)
    red = red[x_borders.min():x_borders.max(), y_borders.min():y_borders.max()]
    green = green[x_borders.min():x_borders.max(), y_borders.min():y_borders.max()]
    blue = blue[x_borders.min():x_borders.max(), y_borders.min():y_borders.max()]
    alpha = alpha[x_borders.min():x_borders.max(), y_borders.min():y_borders.max()]
    reduced = np.where((red == 255) & (green == 255) & (blue == 255))
    alpha[reduced] = 0

    return red, green, blue, alpha


def get_geotransform(ax, alpha):
    """
    Args:
        ax: matplotlib axis that is to be exported
        alpha: the alpha channel of the tiff

    Returns: geotransform tuple, number of rows in image, number of columns in image

    """
    xmin, xmax = tuple(ax.xaxis.get_data_interval())
    ymin, ymax = tuple(ax.yaxis.get_data_interval())

    nrows, ncols = alpha.shape
    xres = (xmax - xmin) / float(ncols)
    yres = (ymax - ymin) / float(nrows)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    return geotransform, nrows, ncols
